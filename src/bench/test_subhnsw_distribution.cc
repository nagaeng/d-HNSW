// test_subhnsw_distribution.cc
// Simple test to analyze the distribution of subhnsw accesses in batches
// Output: CSV for heatmap (batch_id, partition_id, count)

#include <gflags/gflags.h>
#include <grpcpp/grpcpp.h>
#include "../generated/dhnsw.grpc.pb.h"

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <set>
#include <algorithm>
#include <fstream>
#include <numeric>

#include "../dhnsw/DistributedHnsw.h"
#include "../util/read_dataset.h"

typedef int64_t dhnsw_idx_t;

DEFINE_string(server_address, "localhost:50051", "Address of the gRPC server.");
DEFINE_string(query_data_path, "../datasets/sift/sift_query.fvecs", "Path to the query data.");
DEFINE_int32(batch_size, 100, "Batch size for testing.");

using namespace std;

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // --- Read dataset ---
    std::vector<float> query_data;
    int dim_query_data = 0;
    int n_query_data   = 0;

    std::cout << "Reading query data from: " << FLAGS_query_data_path << std::endl;
    query_data = read_fvecs(FLAGS_query_data_path, dim_query_data, n_query_data);
    if (n_query_data <= 0 || dim_query_data <= 0) {
        std::cerr << "Failed to load queries. n=" << n_query_data
                  << " dim=" << dim_query_data << std::endl;
        return 1;
    }
    std::cout << "Loaded " << n_query_data << " queries with dimension " << dim_query_data << std::endl;

    // --- Initialize LocalHnsw (for sift1M) ---
    const int dim = 128;
    const int num_sub_hnsw = 200;
    const int meta_hnsw_neighbors = 16;
    const int sub_hnsw_neighbors = 32;
    const int branching_k = 3;  // Number of subhnsw to search per query

    std::cout << "\nInitializing LocalHnsw..." << std::endl;
    std::cout << "  dim: " << dim << std::endl;
    std::cout << "  num_sub_hnsw: " << num_sub_hnsw << std::endl;
    std::cout << "  branching_k: " << branching_k << std::endl;

    DhnswClient* dhnsw_client = new DhnswClient(
        grpc::CreateChannel(FLAGS_server_address, grpc::InsecureChannelCredentials())
    );
    LocalHnsw local_hnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, dhnsw_client);
    local_hnsw.init();
    std::cout << "LocalHnsw initialized successfully!" << std::endl;

    // --- Allocate arrays for batch processing ---
    const int batch_size = FLAGS_batch_size;
    float* batch_meta_distances = new float[branching_k * batch_size];
    dhnsw_idx_t* batch_meta_labels = new dhnsw_idx_t[branching_k * batch_size];

    // Calculate number of batches needed to cover all queries
    const int num_batches = (n_query_data + batch_size - 1) / batch_size;  // Round up

    // --- Statistics collection ---
    // Per-batch access counts: batch -> (subhnsw_idx -> count)
    std::vector<std::unordered_map<int,int>> per_batch_counts;
    per_batch_counts.reserve(num_batches);

    // Global aggregate (for optional console summary)
    std::unordered_map<int,int> global_subhnsw_access_count;

    std::cout << "\n=== Running meta_search on ALL queries ===" << std::endl;
    std::cout << "Total queries: " << n_query_data << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Number of batches: " << num_batches << std::endl;

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        const int start_query = batch_idx * batch_size;
        const int current_batch_size = std::min(batch_size, n_query_data - start_query);

        const float* batch_query_data_ptr = query_data.data() + start_query * dim_query_data;

        // --- Meta-search for the batch ---
        std::vector<int> sub_hnsw_tosearch_batch;
        local_hnsw.meta_search(
            current_batch_size,
            batch_query_data_ptr,
            branching_k,
            batch_meta_distances,
            batch_meta_labels,
            sub_hnsw_tosearch_batch
        );

        // --- Count accesses for this batch ---
        std::unordered_map<int,int> batch_subhnsw_count;
        batch_subhnsw_count.reserve(256);

        for (int subhnsw_idx : sub_hnsw_tosearch_batch) {
            ++batch_subhnsw_count[subhnsw_idx];
            ++global_subhnsw_access_count[subhnsw_idx];
        }

        per_batch_counts.push_back(std::move(batch_subhnsw_count));

        if ((batch_idx + 1) % 10 == 0) {
            std::cout << "Processed " << (batch_idx + 1) << " / " << num_batches << " batches..." << std::endl;
        }
    }

    // --- Optional: print a brief global summary ---
    {
        const long long total_accesses = 1LL * n_query_data * branching_k;
        std::vector<std::pair<int,int>> sorted_stats(global_subhnsw_access_count.begin(),
                                                     global_subhnsw_access_count.end());
        std::sort(sorted_stats.begin(), sorted_stats.end(),
                  [](const auto& a, const auto& b){ return a.second > b.second; });

        std::cout << "\n=== GLOBAL STATISTICS ===" << std::endl;
        std::cout << "Total queries processed: " << n_query_data << std::endl;
        std::cout << "SubHNSW accesses per query (branching_k): " << branching_k << std::endl;
        std::cout << "Total SubHNSW accesses: " << total_accesses << std::endl;
        std::cout << "Top 10 SubHNSW by access count (index,count,percentage):" << std::endl;

        for (size_t i = 0; i < std::min<size_t>(10, sorted_stats.size()); ++i) {
            double pct = 100.0 * sorted_stats[i].second / double(total_accesses);
            std::cout << "  " << sorted_stats[i].first << ", "
                               << sorted_stats[i].second << ", "
                               << pct << "%\n";
        }
    }

    // --- Export heatmap CSV (long format): batch_id,partition_id,count ---
    {
        // Gather full set of partition_ids observed
        std::set<int> all_parts;
        for (const auto& m : per_batch_counts) {
            for (const auto& kv : m) all_parts.insert(kv.first);
        }

        // If nothing observed, still seed with [0..num_sub_hnsw-1]
        if (all_parts.empty()) {
            for (int pid = 0; pid < num_sub_hnsw; ++pid) all_parts.insert(pid);
        }

        std::string csv_file = "../benchs/subhnsw_heatmap_counts_batch" +
                               std::to_string(batch_size) + ".csv";
        std::ofstream csv(csv_file);
        if (!csv) {
            std::cerr << "Failed to open output file: " << csv_file << std::endl;
        } else {
            csv << "batch_id,partition_id,count\n";
            for (int b = 0; b < (int)per_batch_counts.size(); ++b) {
                for (int pid : all_parts) {
                    int c = 0;
                    auto it = per_batch_counts[b].find(pid);
                    if (it != per_batch_counts[b].end()) c = it->second;
                    csv << b << "," << pid << "," << c << "\n";
                }
            }
            csv.close();
            std::cout << "Heatmap counts CSV saved to: " << csv_file << std::endl;
        }
    }

    // --- Cleanup ---
    delete[] batch_meta_distances;
    delete[] batch_meta_labels;
    delete dhnsw_client;

    return 0;
}