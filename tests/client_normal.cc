// client.cc

#include <gflags/gflags.h>
#include <grpcpp/grpcpp.h>
#include "../generated/dhnsw.grpc.pb.h"
#include "google/protobuf/empty.pb.h"

#include "../../deps/rlibv2/core/lib.hh"
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <chrono>
#include "../dhnsw/DistributedHnsw.h"
#include "../util/read_dataset.h"

DEFINE_string(server_address, "130.127.134.68:50051", "Address of the gRPC server.");
DEFINE_string(rdma_server_address, "192.168.1.2:8888", "Address of the RDMA server.");
DEFINE_int32(use_nic_idx, 3, "Which NIC to create QP");
// 0 for r320 3 for r650
DEFINE_int32(reg_nic_name, 73, "The name to register an opened NIC at rctrl.");
DEFINE_int32(reg_mem_name, 73, "The name to register an MR at rctrl.");
DEFINE_string(query_data_path, "../datasets/sift/sift_query.fvecs", "Path to the query data.");
DEFINE_string(ground_truth_path, "../datasets/sift/sift_groundtruth.ivecs", "Path to the ground truth data.");
// DEFINE_string(query_data_path, "../datasets/gist/gist_query.fvecs", "Path to the query data.");
// DEFINE_string(ground_truth_path, "../datasets/gist/gist_groundtruth.ivecs", "Path to the ground truth data.");

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

    // Initialize RDMA resources
    auto nic = RNic::create(RNicInfo::query_dev_names().at(FLAGS_use_nic_idx)).value();
    auto qp = RC::create(nic, QPConfig()).value();
    // Create a UDP socket to communicate with the server's controller
    ConnectManager cm(FLAGS_rdma_server_address);
    if (cm.wait_ready(1000000, 2) == IOCode::Timeout)
        RDMA_ASSERT(false) << "cm connect to server timeout";

    // Create the pair QP at server using CM
    auto qp_res = cm.cc_rc("client-qp-2-naive", qp, FLAGS_reg_nic_name, QPConfig());
    RDMA_ASSERT(qp_res == IOCode::Ok) << std::get<0>(qp_res.desc);
    auto key = std::get<1>(qp_res.desc);
    RDMA_LOG(4) << "client fetch QP authentical key: " << key;


    // Register local memory
    size_t fixed_size = 1024 * 1024 * 1024; // Adjust size as needed
    auto local_mem = Arc<RMem>(new RMem(fixed_size));
    auto local_mr = RegHandler::create(local_mem, nic).value();
    // Fetch the server's MR info
    auto fetch_res = cm.fetch_remote_mr(FLAGS_reg_mem_name);
    RDMA_ASSERT(fetch_res == IOCode::Ok) << std::get<0>(fetch_res.desc);
    rmem::RegAttr remote_attr = std::get<1>(fetch_res.desc);

    // qp->bind_remote_mr(remote_attr);
    // qp->bind_local_mr(local_mr->get_reg_attr().value());

    RDMA_LOG(4) << "RDMA resources initialized";

    // Connect to the gRPC server to get meta_hnsw, offset, mapping and part
    DhnswClient dhnsw_client(grpc::CreateChannel(FLAGS_server_address, grpc::InsecureChannelCredentials())); 
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
    // std::cout << dim_query_data << std::endl;
    // std::cout << dim_ground_truth << std::endl;
    // std::cout << n_query_data_tmp << std::endl;
    // std::cout << n_ground_truth << std::endl;
    // std::cout << "Read data successfully!" << std::endl;
    // std::cout << "Read query data successfully!" << std::endl;
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
    // std::cout << "Sampled query data successfully!" << std::endl; 
    int branching_k = 10;
    int top_k = 1;

    
    std::vector<int> sub_hnsw_tosearch;

    local_hnsw.set_rdma_qp(qp.get(), remote_attr, local_mr);
    local_hnsw.set_remote_attr(remote_attr);
    local_hnsw.set_local_mr(local_mr, local_mem);

    // Perform meta search
    std::vector<int> ef_search_values = {256, 1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128}; 
    // std::vector<int> ef_search_values = {128}; 
    std::vector<float> recalls;
    std::vector<float> latencies;
    for (int ef : ef_search_values) {
        float* meta_distances = new float[branching_k * n_query_data];
        dhnsw_idx_t* meta_labels = new dhnsw_idx_t[branching_k * n_query_data];
        local_hnsw.set_meta_ef_search(ef);
        std::vector<int> local_sub_hnsw_tag;
        local_hnsw.set_local_sub_hnsw_tag(local_sub_hnsw_tag);

        // Perform sub-searches, fetching sub_hnsw over RDMA as needed
        dhnsw_idx_t* sub_hnsw_tags = new dhnsw_idx_t[top_k * n_query_data];
        dhnsw_idx_t* labels = new dhnsw_idx_t[top_k * n_query_data];
        float* distances = new float[top_k * n_query_data];
        auto start = high_resolution_clock::now();
        local_hnsw.meta_search(n_query_data, query_data.data(), branching_k, meta_distances, meta_labels, sub_hnsw_tosearch);
        local_hnsw.sub_search_each_naive(n_query_data, query_data.data(), branching_k, top_k, distances, labels, sub_hnsw_tosearch, sub_hnsw_tags, ef, RDMA);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<std::chrono::microseconds>(stop - start);
        if(ef != 256){
        latencies.push_back(duration.count());
        }
        std::cout << "Time taken by faiss::IndexHNSWFlat: "
            << duration.count() << " microseconds" << std::endl;
        std::vector<std::vector<dhnsw_idx_t>> mapping = local_hnsw.get_local_mapping(); // for easy testing
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
        // std::cout << "Recall: " << recall << std::endl;
        if(ef != 256){
        recalls.push_back(recall);
        }
        // Clean up
        delete[] labels;
        delete[] distances;
        delete[] sub_hnsw_tags;
        delete[] meta_distances;
        delete[] meta_labels;
    }
    //output the results
    // std::ofstream outfile("../benchs/dhnsw/benchmark_results_gist1M@1.txt");
    std::ofstream outfile("../benchs/naive_dhnsw/benchmark_results_sift1M@1.txt");
    // std::ofstream outfile("../benchs/dhnsw_withoutdb/benchmark_results_gift1M@10.txt");
  
    outfile << "latency(s)\trecall" << std::endl;
    for (size_t i = 0; i < latencies.size(); ++i) {
        outfile << "["  << latencies[i] << ",\t" << recalls[i] << "]," << std::endl;
    }
    outfile.close(); 
    // Finally, clean up RDMA resources
    auto del_res = cm.delete_remote_rc("client-qp-2-naive", key);
    RDMA_ASSERT(del_res == IOCode::Ok)
        << "delete remote QP error: " << del_res.desc;

    RDMA_LOG(4) << "client returns";

    return 0;
}