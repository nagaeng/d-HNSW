#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "../../src/dhnsw/DistributedHnsw.h"
#include <iostream>
#include <fstream>
#include "../../util/read_dataset.h"
#include <chrono>
using hnsw_idx_t = int64_t;
using namespace std::chrono;
int main(){
    int dim = 128;
    //remember to change 
    int num_meta = 50;
    int num_sub_hnsw = 20;
    int meta_hnsw_neighbors = 8;
    int sub_hnsw_neighbors = 32;
    int branching_k = 3;
    int top_k = 1;
    std::string base_data_path = "../datasets/siftsmall/siftsmall_base.fvecs";
    std::string query_data_path = "../datasets/siftsmall/siftsmall_query.fvecs";
    std::string ground_truth_path = "../datasets/siftsmall/siftsmall_groundtruth.ivecs";
    // std::string base_data_path = "../datasets/sift/sift_base.fvecs";
    // std::string query_data_path = "../datasets/sift/sift_query.fvecs";
    // std::string ground_truth_path = "../datasets/sift/sift_groundtruth.ivecs";
    std::vector<float> base_data;
    std::vector<float> query_data;
    std::vector<int> ground_truth;
    int dim_base_data;
    int dim_query_data;
    int dim_ground_truth;
    int n_base_data;
    int n_query_data;
    int n_ground_truth;
    int top_k_ground_truth = 100;
    base_data = read_fvecs(base_data_path, dim_base_data, n_base_data);
    query_data = read_fvecs(query_data_path, dim_query_data, n_query_data);
    ground_truth = read_ivecs(ground_truth_path, dim_ground_truth, n_ground_truth);
    std::cout << dim_base_data << std::endl;
    std::cout << dim_query_data << std::endl;
    std::cout << dim_ground_truth << std::endl;
    std::cout << n_base_data << std::endl;
    std::cout << n_query_data << std::endl;
    std::cout << n_ground_truth << std::endl;
    std::cout << "Read data successfully!" << std::endl;
    faiss::IndexHNSWFlat index(dim_base_data, sub_hnsw_neighbors);
    index.add(n_base_data, base_data.data());
    hnsw_idx_t* I = new hnsw_idx_t[top_k * n_query_data];
    float* D = new float[top_k * n_query_data];

    auto start = high_resolution_clock::now();
    index.search(n_query_data, query_data.data(), top_k, D, I);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time taken by faiss::IndexHNSWFlat: "
         << duration.count() << " microseconds" << std::endl;
    
    int correct = 0;
    float recall = 0.0f;

    std::unordered_set<int> ground_truth_set;
    std::unordered_set<int> result_set;

    for(int i = 0; i < n_query_data; i++) {
        ground_truth_set.clear();
        result_set.clear();
        ground_truth_set.insert(ground_truth.begin() + i * top_k_ground_truth, ground_truth.begin() + i * top_k_ground_truth + top_k);
        result_set.insert(I + i * top_k, I + (i + 1) * top_k);
        for(int j = 0; j < top_k; j++) {
            if(result_set.find(*ground_truth_set.begin()) != result_set.end()) {
                correct++;
            }
        }
    }
    recall = (float)correct / (n_query_data * top_k);
    std::cout << "Recall: " << recall << std::endl;

    return 0;
}

