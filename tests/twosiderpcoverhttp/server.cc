// server.cc

#include <gflags/gflags.h>
#include <grpcpp/grpcpp.h>
#include "dhnsw.grpc.pb.h"


#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>

#include "../dhnsw/DistributedHnsw.h"
#include "../util/read_dataset.h"

DEFINE_int32(port, 50051, "Port for the gRPC server to listen on.");
DEFINE_string(dataset_path, "../datasets/siftsmall/siftsmall_base.fvecs", "Path to the dataset.");

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
    int num_meta = 50;
    int num_sub_hnsw = 20;
    int meta_hnsw_neighbors = 8;
    int sub_hnsw_neighbors = 32;

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
    std::string server_address = "128.110.96.106:" + std::to_string(FLAGS_port);
    std::thread grpc_server_thread(RunGrpcServer, server_address, serialized_meta_hnsw, offset, mapping, part, serialized_data);

    // The main thread can handle RDMA connections or wait
    grpc_server_thread.join();

    return 0;
}