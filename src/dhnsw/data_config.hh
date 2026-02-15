#pragma once

#include <gflags/gflags.h>
#include <string>
#include <vector>
#include <map>
#include <cassert>
#include <iostream>

namespace dhnsw {

DEFINE_string(dataset, "sift1M", "Dataset name: e.g., sift1M, gist1M, deep1B");

struct DatasetConfig {
  std::string query_data_path;
  std::string ground_truth_path;
  int dim;
  int num_sub_hnsw;
  int meta_hnsw_neighbors;
  int sub_hnsw_neighbors;
  std::vector<int> ef_search_values;
  int sampling_mod;
  int sampling_count;
  int num_reps;
  int num_threads;
  int top_k;
  int batch_size;
};

// Global config instance
inline DatasetConfig GlobalDatasetConfig;

inline void load_dataset_config() {
  static std::map<std::string, DatasetConfig> config_map = {
   { "sift1M", {
        "../datasets/sift/sift_query.fvecs",
        "../datasets/sift/sift_groundtruth.ivecs",
        128, 160, 32, 48,
        {48, 48, 48}, 3, 0, 10, 10, 1,5000
    }}, 
    { "gist1M", {
        "../datasets/gist/gist_query.fvecs",
        "../datasets/gist/gist_groundtruth.ivecs",
        960, 120, 32, 48,
        {48,48,48}, 3, 0, 10, 10, 1,5000
    }},
    { "sift10M", {
    "../datasets/sift10M/bigann_query.bvecs",
    "../datasets/sift10M/gnd/idx_10M.ivecs",
    128, 200, 32, 48,
    {48,48,48}, 7, 0, 100, 10, 1,5000
  }},
    { "deep10M", {
        "../datasets/deep100M/deep100M_query.fvecs",
        "../datasets/deep100M/deep10M_groundtruth.ivecs",
        96, 200, 32, 48,
        {48,48,48}, 3, 0, 100, 10, 1,5000
    }},
    { "sift100M", {
      "../datasets/sift10M/bigann_query.bvecs",
      "../datasets/sift10M/gnd/idx_100M.ivecs",
      128, 250, 48, 72,
      {48,48,48}, 3, 0, 100, 10, 1,5000
    }},
  };

  auto it = config_map.find(FLAGS_dataset);
  if (it == config_map.end()) {
    std::cout << "Unsupported dataset name: " << FLAGS_dataset << std::endl;
    exit(1);
  }

  GlobalDatasetConfig = it->second;
}

}  // namespace dhnsw