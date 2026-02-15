#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "../src/dhnsw/DistributedHnsw.h"
#include <iostream>
#include <fstream>
#include "../util/read_dataset.h"
using dhnsw_idx_t = int64_t;
int main(){
    int dim = 128;
    //remember to change 
    int num_data = 10000;
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
    DistributedHnsw dhnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, num_meta);
    dhnsw.build(base_data);

    float* meta_distances = new float[branching_k * n_query_data];
    dhnsw_idx_t* meta_labels = new dhnsw_idx_t[branching_k * n_query_data];
    std::vector<int> sub_hnsw_tosearch;
    dhnsw_idx_t *original_index = new dhnsw_idx_t[top_k * n_query_data];
    dhnsw.meta_search(n_query_data, query_data.data(), branching_k, meta_distances, meta_labels, sub_hnsw_tosearch);
    delete[] meta_distances;
    delete[] meta_labels;
    std::vector<int> local_sub_hnsw;
    for(int i = 0; i < num_sub_hnsw; i++) {
        local_sub_hnsw.push_back(i);
    }
    std::vector<std::vector<float>> sub_hnsw_data = dhnsw.get_sub_hnsw_data(local_sub_hnsw);
    std::vector<std::vector<dhnsw_idx_t>> mapping = dhnsw.get_mapping();
    std::vector<faiss::IndexHNSWFlat*> local_sub_hnsw_indices = dhnsw.get_sub_hnsw(local_sub_hnsw);
    LocalHnsw localhnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, local_sub_hnsw, dhnsw.get_meta_hnsw(), sub_hnsw_data, mapping, local_sub_hnsw_indices);
    dhnsw_idx_t *sub_hnsw_tags = new dhnsw_idx_t[top_k * n_query_data];
    dhnsw_idx_t *labels = new dhnsw_idx_t[top_k * n_query_data];
    float *distances = new float[top_k * n_query_data];
    localhnsw.sub_search_each(n_query_data, query_data.data(), branching_k, top_k, distances, labels, sub_hnsw_tosearch, sub_hnsw_tags);
    for(int i = 0; i < n_query_data; i++) {
        for(int j = 0; j < top_k; j++) {
            original_index[i * top_k + j] = mapping[sub_hnsw_tags[i * top_k + j]][labels[i * top_k + j]];
        }
    }
    int correct = 0;
    float recall = 0.0f;

    std::unordered_set<int> ground_truth_set;
    std::unordered_set<int> result_set;

    for(int i = 0; i < n_query_data; i++) {
        ground_truth_set.clear();
        result_set.clear();
        ground_truth_set.insert(ground_truth.begin() + i * 100, ground_truth.begin() + i * 100 + top_k);
        result_set.insert(original_index + i * top_k, original_index + (i + 1) * top_k);
        for(int j = 0; j < top_k; j++) {
            if(result_set.find(*ground_truth_set.begin()) != result_set.end()) {
                correct++;
            }
        }
    }
    recall = (float)correct / (n_query_data * top_k);
    std::cout << "Recall: " << recall << std::endl;

    delete[] labels;
    delete[] distances;
    delete[] sub_hnsw_tags;
    delete[] original_index;
    return 0;
}


