// server.cc

#include <gflags/gflags.h>
#include <grpcpp/grpcpp.h>
#include "dhnsw.grpc.pb.h"


#include "../../deps/rlib/core/lib.hh"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>

#include "../dhnsw/DistributedHnsw.h"
#include "../util/read_dataset.h"

DEFINE_int32(port, 50051, "Port for the gRPC server to listen on.");
DEFINE_int32(rdma_port, 8888, "Port for the RDMA control channel.");
DEFINE_int32(use_nic_idx, 3, "Which NIC to create QP");
DEFINE_int32(reg_nic_name, 73, "The name to register an opened NIC at rctrl.");
DEFINE_int32(reg_mem_name, 73, "The name to register an MR at rctrl.");
DEFINE_string(dataset_path, "../datasets/sift/sift_base.fvecs", "Path to the dataset.");

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using dhnsw::DhnswService;
using dhnsw::Empty;
using dhnsw::MetaHnswResponse;
using dhnsw::OffsetResponse;
using dhnsw::MappingResponse;
using dhnsw::MappingEntry;
using dhnsw::PartResponse;
using namespace rdmaio;
using namespace rdmaio::rmem;
using namespace rdmaio::qp;

void RunGrpcServer(const std::string& server_address,
                   const std::vector<uint8_t>& serialized_meta_hnsw,
                   const std::vector<size_t>& offset,
                   const std::vector<std::vector<dhnsw_idx_t>>& mapping,
                   const std::vector<int>& part,
                   const std::vector<uint8_t>& serialized_sub_hnsw) {
    DhnswServiceImpl service(serialized_meta_hnsw, offset, mapping, part, serialized_sub_hnsw);

    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate.
    builder.RegisterService(&service);
    // Finally, assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "gRPC server listening on " << server_address << std::endl;

    server->Wait();
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Build the DistributedHnsw index
    int dim = 128;
    int num_meta = 500;
    int num_sub_hnsw = 80;
    int meta_hnsw_neighbors = 32;
    int sub_hnsw_neighbors = 48;

    std::string base_data_path = FLAGS_dataset_path;
    std::vector<float> base_data;
    int dim_base_data;
    int n_base_data;
    base_data = read_fvecs(base_data_path, dim_base_data, n_base_data);
    std::cout << "Read base data successfully!" << std::endl;

    DistributedHnsw dhnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, num_meta);
    dhnsw.build(base_data);

    // Serialize the meta_hnsw and offset
    std::vector<size_t> offset;
    std::vector<uint8_t> serialized_data = dhnsw.serialize_with_record_with_gap(offset);
    std::vector<uint8_t> serialized_meta_hnsw = dhnsw.serialize_meta_hnsw();
    // Get the mapping
    std::vector<std::vector<dhnsw_idx_t>> mapping = dhnsw.get_mapping();
    std::vector<int> part = dhnsw.get_part();
    // Initialize RDMA resources
    RCtrl ctrl(FLAGS_rdma_port);
    RDMA_LOG(4) << "RDMA server listens at localhost:" << FLAGS_rdma_port;

    // Open the NIC
    auto nic = RNic::create(RNicInfo::query_dev_names().at(FLAGS_use_nic_idx)).value();

    // Register the NIC with name 0 to the ctrl
    RDMA_ASSERT(ctrl.opened_nics.reg(FLAGS_reg_nic_name, nic));

    // Allocate memory for the serialized data
    size_t total_size = serialized_data.size();
    RDMA_LOG(4) << "Allocating memory of size: " << total_size;

    auto mr_memory = Arc<RMem>(new RMem(total_size));
    RDMA_ASSERT(ctrl.registered_mrs.create_then_reg(FLAGS_reg_mem_name, mr_memory, nic));

    auto mr_attr = ctrl.registered_mrs.query(FLAGS_reg_mem_name).value()->get_reg_attr().value();
    uint8_t* reg_mem = reinterpret_cast<uint8_t*>(mr_attr.buf);
    // Copy serialized data into the registered memory after the size
    std::memcpy(reg_mem, serialized_data.data(), total_size);

    // Start the listener thread so that clients can communicate with it
    ctrl.start_daemon();
    std::cout << mr_attr.key << std::endl;
    RDMA_LOG(4) << "RDMA resources initialized";

    // Start the gRPC server in a separate thread
    std::string server_address = "130.127.134.42:" + std::to_string(FLAGS_port);
    std::thread grpc_server_thread(RunGrpcServer, server_address, serialized_meta_hnsw, offset, mapping, part, serialized_data);

    // The main thread can handle RDMA connections or wait
    grpc_server_thread.join();

    return 0;
}