// client.cc

#include <gflags/gflags.h>
#include <grpcpp/grpcpp.h>
#include "../generated/dhnsw.grpc.pb.h"
#include "google/protobuf/empty.pb.h"

#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <chrono>
#include "../dhnsw/DistributedHnsw.h"
#include "../util/read_dataset.h"

DEFINE_string(server_address, "130.127.134.68:50051", "Address of the gRPC server.");
DEFINE_string(query_data_path, "../datasets/sift/sift_query.fvecs", "Path to the query data.");
DEFINE_string(ground_truth_path, "../datasets/sift/sift_groundtruth.ivecs", "Path to the ground truth data.");

using grpc::Channel;
using grpc::ClientContext;
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

using namespace std::chrono;

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    // Create ChannelArguments
    grpc::ChannelArguments channel_args;

    // Set the maximum message sizes
    const int max_message_size = 50 * 1024 * 1024; // 50 MB
    channel_args.SetMaxReceiveMessageSize(max_message_size);
    channel_args.SetMaxSendMessageSize(max_message_size);

    // Create the channel with the custom arguments
    std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
        FLAGS_server_address,
        grpc::InsecureChannelCredentials(),
        channel_args);

    DhnswClient dhnsw_client(channel); 
    // Initialize LocalHnsw
    int dim = 128;
    int num_sub_hnsw = 80;
    int meta_hnsw_neighbors = 32;
    int sub_hnsw_neighbors = 48;
    LocalHnsw local_hnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, &dhnsw_client);
    local_hnsw.init();

    // Perform searches
    std::string query_data_path = FLAGS_query_data_path;
    std::string ground_truth_path = FLAGS_ground_truth_path;

    std::vector<float> query_data_tmp;
    std::vector<int> ground_truth_tmp;
    std::vector<float> query_data;
    std::vector<int> ground_truth;
    int dim_query_data;
    int n_query_data_tmp;
    int dim_ground_truth;
    int n_ground_truth;
    query_data_tmp = read_fvecs(query_data_path, dim_query_data, n_query_data_tmp);
    ground_truth_tmp = read_ivecs(ground_truth_path, dim_ground_truth, n_ground_truth);

    int branching_k = 10;
    int top_k = 1;
    for(int i = 0; i < n_query_data_tmp; i++) {
        if(i % 3 == 1){ // sample 1/3 of the query data (remember to change)
            query_data.insert(query_data.end(),
                query_data_tmp.begin() + i * dim_query_data,
                query_data_tmp.begin() + (i + 1) * dim_query_data);

            ground_truth.insert(ground_truth.end(),
                ground_truth_tmp.begin() + i * dim_ground_truth,
                ground_truth_tmp.begin() + (i + 1) * dim_ground_truth);
        }
    }
    int n_query_data = query_data.size() / dim_query_data;
    float* meta_distances = new float[branching_k * n_query_data];
    dhnsw_idx_t* meta_labels = new dhnsw_idx_t[branching_k * n_query_data];
    std::vector<int> sub_hnsw_tosearch;
    std::vector<int> ef_search_values = {256, 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128}; 
    // std::vector<int> ef_search_values = {128}; 
    std::vector<float> recalls;
    std::vector<float> latencies;
    // Perform meta search
    for (int ef : ef_search_values) {
    auto start = high_resolution_clock::now();
    local_hnsw.meta_search(n_query_data, query_data.data(), branching_k, meta_distances, meta_labels, sub_hnsw_tosearch);
    // std::cout << "Meta search completed!" << std::endl;
    // std::cout << "sub hnsw to search: ";
    // Set up local_sub_hnsw_tag (initially empty)
    std::vector<int> local_sub_hnsw_tag;
    local_hnsw.set_local_sub_hnsw_tag(local_sub_hnsw_tag);

    // Perform sub-searches, fetching sub_hnsw over RDMA as needed
    dhnsw_idx_t* sub_hnsw_tags = new dhnsw_idx_t[top_k * n_query_data];
    dhnsw_idx_t* labels = new dhnsw_idx_t[top_k * n_query_data];
    float* distances = new float[top_k * n_query_data];
    local_hnsw.sub_search_each_naive(n_query_data, query_data.data(), branching_k, top_k, distances, labels, sub_hnsw_tosearch, sub_hnsw_tags, ef, fetch_type::RPC);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::microseconds>(stop - start);
    std::vector<std::vector<dhnsw_idx_t>> mapping = local_hnsw.get_local_mapping(); // for easy testing
    if(ef != 256){
        latencies.push_back(duration.count());
    }
    dhnsw_idx_t *original_index = new dhnsw_idx_t[top_k * n_query_data];
    for(int i = 0; i < n_query_data; i++) {
        for(int j = 0; j < top_k; j++) {
            original_index[i * top_k + j] = mapping[sub_hnsw_tags[i * top_k + j]][labels[i * top_k + j]];
        }
    }
    int correct = 0;
        float recall = 0.0f;
        for(int i = 0; i < n_query_data; i++) {
            std::unordered_set<int> ground_truth_set(
                ground_truth.begin() + i * dim_ground_truth,
                ground_truth.begin() + i * dim_ground_truth + top_k
            );

            for(int j = 0; j < top_k; j++) {
                int retrieved_idx = original_index[i * top_k + j];
                if (ground_truth_set.find(retrieved_idx) != ground_truth_set.end()) {
                    correct++;
                }
            }
        }

    recall = static_cast<float>(correct) / (n_query_data * top_k); 
    std::cout << "Recall: " << recall << std::endl;
    // Clean up
    if(ef != 256){
        recalls.push_back(recall);
        }
    delete[] labels;
    delete[] distances;
    delete[] sub_hnsw_tags;
    delete[] meta_distances;
    delete[] meta_labels;
    }
    std::ofstream outfile("../benchs/twosiderpc/benchmark_results_sift1M@1.txt");
    // std::ofstream outfile("../benchs/dhnsw_withoutdb/benchmark_results_gift1M@10.txt");
  
    outfile << "latency(s)\trecall" << std::endl;
    for (size_t i = 0; i < latencies.size(); ++i) {
        outfile << "["  << latencies[i] << ",\t" << recalls[i] << "]," << std::endl;
    }
    outfile.close(); 
    return 0;
}