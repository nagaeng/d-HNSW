// server.cc

#include <gflags/gflags.h>
#include <grpcpp/grpcpp.h>
#include "../generated/dhnsw.grpc.pb.h"

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
DEFINE_string(server_ip, "0.0.0.0", "IP address for the gRPC server to bind.");
DEFINE_int32(dim, 128, "Vector dimension.");
DEFINE_int32(num_meta, 5000, "Number of meta centroids.");
DEFINE_int32(num_sub_hnsw, 160, "Number of sub-HNSW partitions.");
DEFINE_int32(meta_hnsw_neighbors, 32, "Meta-HNSW graph degree.");
DEFINE_int32(sub_hnsw_neighbors, 48, "Sub-HNSW graph degree.");
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using dhnsw::DhnswService;
using dhnsw::Empty;
using dhnsw::MetaHnswResponse;
using dhnsw::MappingResponse;
using dhnsw::MappingEntry;
using dhnsw::OverflowResponse;
using dhnsw::Offset_SubHnswResponse;
using dhnsw::Offset_ParaResponse;
using namespace rdmaio;
using namespace rdmaio::rmem;
using namespace rdmaio::qp;

void RunGrpcServer(const std::string& server_address,
                   const std::vector<uint8_t>& serialized_meta_hnsw,
                   const std::vector<uint64_t>& offset_sub_hnsw,
                   const std::vector<uint64_t>& offset_para,
                   const std::vector<uint64_t>& overflow,
                   const std::vector<std::vector<dhnsw_idx_t>>& mapping) {
    DhnswServiceImpl service(serialized_meta_hnsw, offset_sub_hnsw, offset_para, overflow, mapping);

    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    
    builder.SetMaxReceiveMessageSize(INT_MAX);
    builder.SetMaxSendMessageSize(INT_MAX);
    
    // Register "service" as the instance through which we'll communicate.
    builder.RegisterService(&service);
    // Finally, assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "gRPC server listening on " << server_address << std::endl;

    server->Wait();
}

void bind_thread_to_cores(int thread_id, int core_start, int cores_per_thread) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    // Set affinity to assigned cores
    for (int i = 0; i < cores_per_thread; i++) {
        int core_id = core_start + i;
        CPU_SET(core_id, &cpuset);
    }
    
    // Apply the CPU affinity mask to the current thread
    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error: unable to set thread affinity for thread " << thread_id 
                  << " to cores " << core_start << "-" << (core_start + cores_per_thread - 1) 
                  << ", error code: " << rc << std::endl;
    } else {
        std::cout << "Thread " << thread_id << " bound to cores " 
                  << core_start << "-" << (core_start + cores_per_thread - 1) << std::endl;
    }
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    bind_thread_to_cores(0, 0, 144);
    omp_set_num_threads(144);
    // Build the DistributedHnsw index
    int dim = FLAGS_dim;
    int num_meta = FLAGS_num_meta;
    int num_sub_hnsw = FLAGS_num_sub_hnsw;
    int meta_hnsw_neighbors = FLAGS_meta_hnsw_neighbors;
    int sub_hnsw_neighbors = FLAGS_sub_hnsw_neighbors;

    // std::string base_data_path = FLAGS_dataset_path;
    // std::vector<float> base_data;
    // int dim_base_data;
    // int n_base_data;
    // read sift1M or gist1M 
    // base_data = read_fvecs(base_data_path, dim_base_data, n_base_data);
    // read sift10M
    // base_data = read_bvecs(base_data_path, dim_base_data, n_base_data, 1, 10000000);
    // read deep1B
    // base_data = read_fvecs(base_data_path, dim_base_data, n_base_data);
    // std::cout << "Read base data successfully!" << std::endl;

    DistributedHnsw dhnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, num_meta);
    // dhnsw.build(base_data, 1000);
    // for (int i = 1; i <= 100000000; i += 10000000) {
    //     int dim, num;
    //     auto base_data = read_fvecs(FLAGS_dataset_path, dim, num, i, i + 10000000 - 1);
    //     if( i == 1) {
    //         std::cout << "Batch size: " << base_data.size() << std::endl;
    //         dhnsw.build(base_data, 1000);
    //     }
    //     else {
    //         dhnsw.insert(10000000, base_data);
    //     }
        
    // }
    int dim_, num_;
    auto base_data = read_fvecs(FLAGS_dataset_path, dim_, num_);
    std::cout << "Batch size: " << base_data.size() << std::endl;
    dhnsw.build(base_data, 1000);
    //print size of meta_hnsw
    std::cout << "Size of meta_hnsw: " << dhnsw.get_meta_hnsw_size() << " MB" << std::endl;
    std::cout << "Thread " << std::max((size_t)10 * 10 * 1000 * 1000 / (dhnsw.sub_hnsw[0]->hnsw.max_level * dim * dhnsw.sub_hnsw[0]->hnsw.efSearch + 1), (size_t)1) << std::endl;
    // Serialize the meta_hnsw and offset
    std::vector<uint64_t> offset_sub_hnsw;
    std::vector<uint64_t> offset_para;
    std::vector<uint64_t> overflow;
    std::vector<uint8_t> serialized_data = dhnsw.serialize_with_record_with_in_out_gap(offset_sub_hnsw, offset_para, overflow);
    std::cout << "Serialized data size: " << serialized_data.size() << " MB" << std::endl;
    std::vector<uint8_t> serialized_meta_hnsw = dhnsw.serialize_meta_hnsw();
    std::cout << "Serialized meta_hnsw size: " << serialized_meta_hnsw.size() << " MB" << std::endl;
    // Get the mapping
    std::vector<std::vector<dhnsw_idx_t>> mapping = dhnsw.get_mapping();
    std::cout << "Mapping size: " << mapping.size() << std::endl;
    // Initialize RDMA resources
    RCtrl ctrl(FLAGS_rdma_port);
    RDMA_LOG(4) << "RDMA server listens at localhost:" << FLAGS_rdma_port;

    // Open the NIC
    auto nic = RNic::create(RNicInfo::query_dev_names().at(FLAGS_use_nic_idx)).value();
    std::cout << "NIC created" << std::endl;
    // Register the NIC with name 0 to the ctrl
    RDMA_ASSERT(ctrl.opened_nics.reg(FLAGS_reg_nic_name, nic));

    // Allocate memory for the serialized data
    size_t total_size = serialized_data.size();
    std::cout << "Total size: " << total_size << std::endl;
    RDMA_LOG(4) << "Allocating memory of size: " << total_size;

    auto mr_memory = Arc<RMem>(new RMem(total_size));
    RDMA_ASSERT(ctrl.registered_mrs.create_then_reg(FLAGS_reg_mem_name, mr_memory, nic));

    auto mr_attr = ctrl.registered_mrs.query(FLAGS_reg_mem_name).value()->get_reg_attr().value();
    std::cout << "MR created" << std::endl;
    uint8_t* reg_mem = reinterpret_cast<uint8_t*>(mr_attr.buf);
    std::cout << "Reg mem created" << std::endl;
    // Copy serialized data into the registered memory after the size
    std::memcpy(reg_mem, serialized_data.data(), total_size);
    std::cout << "Data copied" << std::endl;
    // Start the listener thread so that clients can communicate with it
    ctrl.start_daemon();
    std::cout << mr_attr.key << std::endl;
    RDMA_LOG(4) << "RDMA resources initialized";

    // Start the gRPC server in a separate thread
    std::string server_address = FLAGS_server_ip + ":" + std::to_string(FLAGS_port);
    std::thread grpc_server_thread(RunGrpcServer, server_address, serialized_meta_hnsw, offset_sub_hnsw, offset_para, overflow, mapping);

    // The main thread can handle RDMA connections or wait
    grpc_server_thread.join();

    return 0;
}