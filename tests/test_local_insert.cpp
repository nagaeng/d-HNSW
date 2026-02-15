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
#include <unordered_set>
#include <set>
#include <iterator>

#include "../dhnsw/DistributedHnsw.h"
#include "../util/read_dataset.h"

DEFINE_string(dataset_path, "../datasets/sift/sift_base.fvecs", "Path to the dataset.");
// DEFINE_string(dataset_path, "../datasets/gist/gist_base.fvecs", "Path to the dataset.");
DEFINE_string(query_data_path, "../datasets/sift/sift_query.fvecs", "Path to the query data.");
DEFINE_string(ground_truth_path, "../datasets/sift/sift_groundtruth.ivecs", "Path to the ground truth data.");


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
    int dim = 128;
    int num_meta = 5000; // for sift kmeans
    // int num_sub_hnsw = 120;
    int num_sub_hnsw = 180;
    int meta_hnsw_neighbors = 48;
    int sub_hnsw_neighbors = 72;

    std::string base_data_path = FLAGS_dataset_path;
    std::vector<float> base_data;
    int dim_base_data;
    int n_base_data;
    base_data = read_fvecs(base_data_path, dim_base_data, n_base_data);
    std::cout << "Read base data successfully!" << std::endl;

    DistributedHnsw dhnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, num_meta);
    dhnsw.build(base_data, 12500);
    dhnsw.print_subhnsw_balance();
    //print space used by one sub_hnsw
    std::cout << "Space used by one sub_hnsw: " << dhnsw.get_per_sub_hnsw_size() << std::endl;
    std::cout << "Space used by meta_hnsw: " << dhnsw.get_meta_hnsw_size() << std::endl;
    // search query_data 
    std::vector<float> query_data;
    std::string query_data_path = FLAGS_query_data_path;
    int dim_query_data;
    int n_query_data;
    query_data = read_fvecs(query_data_path, dim_query_data, n_query_data);
    std::vector<float> query_data_tmp;
    query_data_tmp = read_fvecs(query_data_path, dim_query_data, n_query_data);
    std::vector<int> ground_truth_tmp;
    std::string ground_truth_path = FLAGS_ground_truth_path;
    int dim_ground_truth;
    int n_ground_truth;
    ground_truth_tmp = read_ivecs(ground_truth_path, dim_ground_truth, n_ground_truth);

    // test recall top 10
    float recall = 0.0f;
    int top_k = 10;
    std::vector<float> distances(n_query_data * top_k);
    std::vector<dhnsw_idx_t> labels(n_query_data * top_k);
    std::vector<dhnsw_idx_t> sub_hnsw_tags(n_query_data * top_k);
    std::vector<dhnsw_idx_t> original_index(n_query_data * top_k);
    int branching_k = 5;

    int efSearch = 72;
    dhnsw.hierarchicalSearch(n_query_data, query_data.data(), branching_k, top_k, distances.data(), labels.data(), sub_hnsw_tags.data(), original_index.data(), efSearch);

    for (int i = 0; i < n_query_data; i++) {
        std::unordered_set<dhnsw_idx_t> retrieved_set;
        std::unordered_set<dhnsw_idx_t> ground_truth_set;
        
        // Add retrieved results to set
        for (int j = 0; j < top_k; j++) {
            retrieved_set.insert(original_index[i * top_k + j]);
        }
        
        // Add ground truth results to set - each query has 100 ground truth entries
        for (int j = 0; j < top_k; j++) {
            ground_truth_set.insert(ground_truth_tmp[i * 100 + j]);
        }
        
        // Count intersection
        int correct = 0;
        for (const auto& gt_item : ground_truth_set) {
            if (retrieved_set.find(gt_item) != retrieved_set.end()) {
                correct++;
            }
        }
        
        recall += static_cast<float>(correct) / top_k;
    }
    
    // Average recall across all queries
    recall = recall / n_query_data;
    std::cout << "Recall: " << recall << std::endl;
    /* test later
    std::cout << "Thread " << std::max((size_t)10 * 10 * 1000 * 1000 / (dhnsw.sub_hnsw[0]->hnsw.max_level * dim * dhnsw.sub_hnsw[0]->hnsw.efSearch + 1), (size_t)1) << std::endl;
    // Serialize the meta_hnsw and offset
    std::vector<uint64_t> offset_sub_hnsw;
    std::vector<uint64_t> offset_para;
    std::vector<uint64_t> overflow;
    std::vector<uint8_t> serialized_meta_hnsw = dhnsw.initial_serialize_whole(offset_sub_hnsw, offset_para, overflow);
    
    std::cout << "Serialization completed with:" << std::endl;
    std::cout << "  - offset_sub_hnsw size: " << offset_sub_hnsw.size() << std::endl;
    std::cout << "  - offset_para size: " << offset_para.size() << std::endl;
    std::cout << "  - overflow size: " << overflow.size() << std::endl;
    std::cout << "  - serialized data size: " << serialized_meta_hnsw.size() << std::endl;
    
    // Print some values from offset arrays for debugging
    if (!offset_sub_hnsw.empty()) {
        std::cout << "First few offset_sub_hnsw values: ";
        for (size_t i = 0; i < std::min(size_t(4), offset_sub_hnsw.size()); i++) {
            std::cout << offset_sub_hnsw[i] << " ";
        }
        std::cout << std::endl;
    }
    
    DistributedHnsw dhnsw_deserialized = DistributedHnsw::initial_deserialize_whole(serialized_meta_hnsw, offset_sub_hnsw, offset_para, overflow);
    std::cout << "Deserialized meta_hnsw successfully!" << std::endl;
    std::cout << "Number of sub_hnsw indices: " << dhnsw_deserialized.sub_hnsw.size() << std::endl;
    
    // Check if any sub_hnsw pointers are valid
    int valid_count = 0;
    for (size_t i = 0; i < dhnsw_deserialized.sub_hnsw.size(); i++) {
        if (dhnsw_deserialized.sub_hnsw[i] != nullptr) {
            valid_count++;
        }
    }
    std::cout << "Valid sub_hnsw count: " << valid_count << " out of " << dhnsw_deserialized.sub_hnsw.size() << std::endl;
    
    if (!dhnsw_deserialized.sub_hnsw.empty() && dhnsw_deserialized.sub_hnsw[0] != nullptr) {
        std::cout << "sub_hnsw[0] max_level: " << dhnsw_deserialized.sub_hnsw[0]->hnsw.max_level << std::endl;
        std::cout << "sub_hnsw[0] efSearch: " << dhnsw_deserialized.sub_hnsw[0]->hnsw.efSearch << std::endl;
    } else {
        std::cout << "Warning: sub_hnsw[0] is null or empty, cannot display properties" << std::endl;
    }
    // // Get the mapping
    // std::vector<std::vector<dhnsw_idx_t>> mapping = dhnsw.get_mapping();
    // std::vector<int> part = dhnsw.get_part();
    // int dim_query_data;
    // int dim_ground_truth;
    // std::string query_data_path = FLAGS_query_data_path;
    // std::string ground_truth_path = FLAGS_ground_truth_path;

    // std::vector<float> query_data_tmp;
    // std::vector<int> ground_truth_tmp;
    // int n_query_data_tmp;
    // int n_ground_truth_tmp;
    // query_data_tmp = read_fvecs(query_data_path, dim_query_data, n_query_data_tmp);
    // ground_truth_tmp = read_ivecs(ground_truth_path, dim_ground_truth, n_ground_truth_tmp);

    // Insert data

    return 0;
    */
}