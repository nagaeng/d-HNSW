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
// DEFINE_string(dataset_path, "../datasets/sift/sift_base.fvecs", "Path to the dataset.");
// DEFINE_string(dataset_path, "../datasets/gist/gist_base.fvecs", "Path to the dataset.");
// DEFINE_string(dataset_path, "../datasets/sift10M/bigann_base.bvecs", "Path to the dataset.");
// DEFINE_string(dataset_path, "../datasets/deep1B/deep1B_base.fvecs", "Path to the dataset.");
DEFINE_string(dataset_path, "../datasets/deep100M/deep100M_base.fvecs", "Path to the dataset.");

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

    builder.SetMaxReceiveMessageSize(INT_MAX);
    builder.SetMaxSendMessageSize(INT_MAX);
    // Register "service" as the instance through which we'll communicate.
    builder.RegisterService(&service);
    // Finally, assemble the server.
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "gRPC server listening on " << server_address << std::endl;

    server->Wait();
}

using hnsw_idx_t = int64_t;

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
    bind_thread_to_cores(0, 0, 144);
    omp_set_num_threads(144);
    // Build the DistributedHnsw index
    // int dim = 960;
    // int dim = 128;
    int dim = 96;
    int num_meta = 50000;
    int num_sub_hnsw = 100;
    // int num_sub_hnsw = 120;
    int meta_hnsw_neighbors = 72;
    int sub_hnsw_neighbors = 128;

    std::string query_data_path = "../datasets/deep100M/deep100M_query.fvecs";
    std::string ground_truth_path = "../datasets/deep100M/deep100M_groundtruth.ivecs";

    DistributedHnsw dhnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, num_meta); 
    {
    int dim_, num_;
    auto base_data = read_fvecs(FLAGS_dataset_path, dim_, num_);
    std::cout << "Batch size: " << base_data.size() << std::endl;
    dhnsw.build(base_data, 1000);
    }
    int n_query_data;
    int n_ground_truth;
    int top_k_ground_truth = 100;
    std::vector<float> query_data;
    std::vector<int> ground_truth;
    int dim_query_data;
    int dim_ground_truth;
    int top_k = 1;
    query_data = read_fvecs(query_data_path, dim_query_data, n_query_data);
    ground_truth = read_ivecs(ground_truth_path, dim_ground_truth, n_ground_truth);
    hnsw_idx_t* I = new hnsw_idx_t[top_k * n_query_data];
    float* D = new float[top_k * n_query_data];
    hnsw_idx_t* sub_hnsw_tags = new hnsw_idx_t[top_k * n_query_data];
    hnsw_idx_t* original_index = new hnsw_idx_t[top_k * n_query_data];
    
    dhnsw.hierarchicalSearch(n_query_data, query_data.data(), 5, 1, D, I, sub_hnsw_tags, original_index, 512);
    int correct = 0;
    float recall = 0.0f;

    std::unordered_set<int> ground_truth_set;
    std::unordered_set<int> result_set;

    for(int i = 0; i < n_query_data; i++) {
        ground_truth_set.clear();
        result_set.clear();
        ground_truth_set.insert(ground_truth.begin() + i * top_k_ground_truth, ground_truth.begin() + i * top_k_ground_truth + top_k);
        result_set.insert(original_index + i * top_k, original_index + (i + 1) * top_k);
        for(int j = 0; j < top_k; j++) {
            if(result_set.find(*ground_truth_set.begin()) != result_set.end()) {
                correct++;
            }
        }
    }
    recall = (float)correct / (n_query_data * top_k);
    std::cout << "Recall: " << recall << std::endl;
    
    // Clean up allocated memory
    delete[] I;
    delete[] D;
    delete[] sub_hnsw_tags;
    delete[] original_index;
}