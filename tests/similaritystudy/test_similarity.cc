#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "../dhnsw/DistributedHnsw.h"
#include <iostream>
#include <fstream>
#include "../util/read_dataset.h"
#include <unordered_set>
#include <chrono>

using dhnsw_idx_t = int64_t;
int main(){
    omp_set_num_threads(144);
    int dim = 128;
    //remember to change 
    int num_data = 1000000;
    int num_meta = 500;
    int num_sub_hnsw = 40;
    int meta_hnsw_neighbors = 8;
    int sub_hnsw_neighbors = 32;
    int branching_k = 5;
    int top_k = 1;
    // std::string base_data_path = "../datasets/siftsmall/siftsmall_base.fvecs";
    // std::string query_data_path = "../datasets/siftsmall/siftsmall_query.fvecs";
    // std::string ground_truth_path = "../datasets/siftsmall/siftsmall_groundtruth.ivecs";
    std::string base_data_path = "../datasets/sift/sift_base.fvecs";
    std::string query_data_path = "../datasets/sift/sift_query.fvecs";
    std::string ground_truth_path = "../datasets/sift/sift_groundtruth.ivecs";
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

    std::vector<int> ef_search_values = {1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128};

    // Store results for later analysis
    std::vector<float> recalls;
    std::vector<double> latencies;

    // Output header for results
    std::cout << "\nTesting recall vs latency:" << std::endl;
    std::cout << "efSearch, Recall, Latency (ms)" << std::endl;

    // Prepare for searching
    int K_meta = branching_k;
    int K_sub = top_k;

    for (int efSearch : ef_search_values) {
        float* meta_distances = new float[branching_k * n_query_data];
        dhnsw_idx_t* meta_labels = new dhnsw_idx_t[branching_k * n_query_data];
        dhnsw_idx_t* sub_hnsw_tags = new dhnsw_idx_t[top_k * n_query_data];
        dhnsw_idx_t* labels = new dhnsw_idx_t[top_k * n_query_data];
        float* distances = new float[top_k * n_query_data];
        
        int correct = 0;
        float recall = 0.0f;
        std::vector<int> sub_hnsw_tosearch;
        // Measure search time
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Perform hierarchical search with current efSearch value
        dhnsw.meta_search(n_query_data, query_data.data(), K_meta, 
                                 meta_distances, meta_labels, sub_hnsw_tosearch);
        std::cout << "Meta search done" << std::endl;
        for(int i = 0; i < n_query_data; i++) {
            std::vector<std::tuple<float, dhnsw_idx_t, dhnsw_idx_t>> result;
            for(int j = 0; j < K_meta; j++) {
                int sub_idx = sub_hnsw_tosearch[i * K_meta + j];
                float tmp_sub_distances[K_sub];
                dhnsw_idx_t tmp_sub_labels[K_sub];
                dhnsw.sub_hnsw[sub_idx]->hnsw.efSearch = efSearch;
                dhnsw.sub_hnsw[sub_idx]->search(1, query_data.data() + i * dim, K_sub, tmp_sub_distances, tmp_sub_labels);

                // Collect results
                for (int k = 0; k < K_sub; k++) {
                    result.emplace_back(tmp_sub_distances[k], tmp_sub_labels[k], sub_idx);
                }
            }
            std::sort(result.begin(), result.end());
            for (int k = 0; k < K_sub && k < result.size(); k++) {
                distances[i * K_sub + k] = std::get<0>(result[k]);
                labels[i * K_sub + k] = std::get<1>(result[k]);
                sub_hnsw_tags[i * K_sub + k] = std::get<2>(result[k]);
        }
        }
        std::cout << "Sub search done" << std::endl;
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double latency_ms = duration.count();
        std::cout << "Latency: " << latency_ms << " ms" << std::endl;
        // Calculate recall
        std::unordered_set<int> ground_truth_set;
        std::unordered_set<int> result_set;

        for(int i = 0; i < n_query_data; i++) {
            ground_truth_set.clear();
            result_set.clear();
            ground_truth_set.insert(ground_truth.begin() + i * 100, ground_truth.begin() + i * 100 + top_k);
            result_set.insert(labels + i * top_k, labels + (i + 1) * top_k);
            for(int j = 0; j < top_k; j++) {
                if(result_set.find(*ground_truth_set.begin()) != result_set.end()) {
                    correct++;
                }
            }
        }
        recall = (float)correct / (n_query_data * top_k);
        
        // Store results
        recalls.push_back(recall);
        latencies.push_back(latency_ms);
        
        // Output results for this efSearch value
        std::cout <<  "[" << latency_ms << ", " << recall <<"],"<< std::endl;
        
        // Clean up
        delete[] labels;
        delete[] distances;
        delete[] sub_hnsw_tags;
        delete[] meta_distances;
        delete[] meta_labels;
    }

    // Output summary
    std::cout << "\nSummary of results:" << std::endl;
    for (size_t i = 0; i < ef_search_values.size(); i++) {
        std::cout << "efSearch: " << ef_search_values[i] 
                  << ", Recall: " << recalls[i] 
                  << ", Latency: " << latencies[i] << " ms" << std::endl;
    }

    return 0;
}


