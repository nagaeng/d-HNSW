#include <iostream>
#include <vector>
#include <cassert>
#include "../src/dhnsw/DistributedHnsw.h"
#include <iostream>
#include <fstream>
#include <vector>
using dhnsw_idx_t = int64_t;
void testBuild() {
    int dim = 64;
    int num_data = 10000;
    int num_meta = 500;
    int num_sub_hnsw = 10;
    int meta_hnsw_neighbors = 8;
    int sub_hnsw_neighbors = 16;
    int branching_k = 3;
    
    DistributedHnsw dhnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, num_meta);
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    // Generate some random data
    float* data = new float[dim * num_data];
    for (int i = 0; i < num_data; i++) {
        for (int j = 0; j < dim; j++)
            data[dim * i + j] = distrib(rng);
        data[dim * i] += i / 1000.;
    }
    std::vector<float> vec(data, data + dim * num_data);
    // Build the index
    dhnsw.build(vec);

    std::cout << "Build test passed!" << std::endl;

    //search
    int num_queries = 100;
    int top_k = 10;
    float *query = new float[dim * num_queries];
    for (int i = 0; i < num_queries; i++) {
        for (int j = 0; j < dim; j++)
            query[dim * i + j] = distrib(rng);
        query[dim * i] += i / 1000.;
    }
    dhnsw_idx_t *labels = new dhnsw_idx_t[top_k * num_queries];
    float *distances = new float[top_k * num_queries];
    dhnsw_idx_t *sub_hnsw_tags = new dhnsw_idx_t[top_k * num_queries];
    dhnsw_idx_t *original_index = new dhnsw_idx_t[top_k * num_queries]; 
    dhnsw.hierarchicalSearch(num_queries, query, branching_k, top_k, distances, labels, sub_hnsw_tags, original_index);
    for(int i = 0; i < num_queries; i++) {
        std::cout << "Query " << i << std::endl;
        for(int j = 0; j < top_k; j++) {
            std::cout << labels[i * top_k + j] << " ";
            std::cout << sub_hnsw_tags[i * top_k + j] << " ";
            std::cout << original_index[i * top_k + j] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }
    std::cout << "Search test passed!" << std::endl;

    //clean up
    delete[] data;
    delete[] query;
    delete[] labels;
    delete[] distances;
    delete[] sub_hnsw_tags;
    delete[] original_index;
}


int main() {
    testBuild();
    return 0;
}