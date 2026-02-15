#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "../../src/dhnsw/DistributedHnsw.h"
#include <iostream>
#include <fstream>
#include "../../util/read_dataset.h"
#include <chrono>
#include <unordered_set>
#include <faiss/IndexHNSW.h>
using hnsw_idx_t = int64_t;
using namespace std::chrono;
int main(){

    std::string base_data_path = "../datasets/sift/sift_base.fvecs";
    std::string query_data_path = "../datasets/sift/sift_query.fvecs";
    std::string ground_truth_path = "../datasets/sift/sift_groundtruth.ivecs";
   
    // Build the DistributedHnsw index
    int dim = 128;
    int num_meta = 5000;
    int num_sub_hnsw = 160;
    int meta_hnsw_neighbors = 32;
    int sub_hnsw_neighbors = 48;

    std::vector<float> base_data;
    int dim_base_data;
    int n_base_data;
    base_data = read_fvecs(base_data_path, dim_base_data, n_base_data);
    std::cout << "Read base data successfully!" << std::endl;

    DistributedHnsw dhnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, num_meta);
    dhnsw.build(base_data,1000);
    std::cout << "Build the index successfully!" << std::endl;
    std::vector<uint8_t> buffer;
    std::vector<uint64_t> offset_sub_hnsw;
    std::vector<uint64_t> offset_para;
    std::vector<uint64_t> overflow;
    buffer = dhnsw.serialize_with_record_with_in_out_gap(offset_sub_hnsw, offset_para, overflow);
    std::cout << "Serialized the whole index successfully!" << std::endl;
    //print offset_sub_hnsw, offset_para, overflow
    DistributedHnsw dhnsw_deserialized = dhnsw.deserialize_with_record_with_in_out_gap(buffer, offset_sub_hnsw, offset_para, overflow);
    std::cout << "Deserialized the whole index successfully!" << std::endl;
    //check the index
    std::vector<uint8_t> data_sub_hnsw_1( buffer.begin() + offset_sub_hnsw[2], buffer.begin() + offset_sub_hnsw[3]);
    faiss::IndexHNSWFlat* sub_hnsw_1 = dhnsw_deserialized.deserialize_sub_hnsw_with_record_with_in_out_gap(data_sub_hnsw_1, 1, offset_sub_hnsw, offset_para, overflow);
    std::cout << "Deserialized the sub_hnsw[1] successfully!" << std::endl;
    //check the index
    std::vector<uint8_t> data_sub_hnsw_2( buffer.begin() + offset_sub_hnsw[4], buffer.begin() + offset_sub_hnsw[5]);
    faiss::IndexHNSWFlat* sub_hnsw_2 = dhnsw_deserialized.deserialize_sub_hnsw_with_record_with_in_out_gap(data_sub_hnsw_2, 2, offset_sub_hnsw, offset_para, overflow);
    std::cout << "Deserialized the sub_hnsw[2] successfully!" << std::endl;
    //check the index
    std::vector<uint8_t> data_sub_hnsw_150( buffer.begin() + offset_sub_hnsw[150 * 2], buffer.begin() + offset_sub_hnsw[150 * 2 + 1]);
    faiss::IndexHNSWFlat* sub_hnsw_150 = dhnsw_deserialized.deserialize_sub_hnsw_with_record_with_in_out_gap(data_sub_hnsw_150, 150, offset_sub_hnsw, offset_para, overflow);
    std::cout << "Deserialized the sub_hnsw[150] successfully!" << std::endl;

    return 0;
}


