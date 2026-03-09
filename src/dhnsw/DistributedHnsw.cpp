#include "DistributedHnsw.h"
#include <grpcpp/grpcpp.h>
#include "../generated/dhnsw.grpc.pb.h"
#include "google/protobuf/empty.pb.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/MetaIndexes.h>
#include <faiss/impl/HNSW.h>
#include <faiss/Clustering.h>
#include <faiss/utils/distances.h>
#include <sstream>
// #include <KaHIP/interface/kaHIP_interface.h>
#include <omp.h>
#include <faiss/utils/Heap.h>
#include <mutex>
#include <shared_mutex>
#include <faiss/index_io.h>
#include <faiss/impl/io_macros.h>
#include <faiss/index_io.h>

#include <cstring>
#include <faiss/impl/io.h> 
#include <cstdint> // for uint8_t
#include "../../deps/rlib/core/lib.hh"
#include "../../deps/rlib/core/qps/doorbell_helper.hh"
#include "../../deps/rlib/benchs/bench_op.hh"
#include "../../xcomm/src/rpc/mod.hh"
#include "../../xcomm/src/transport/rdma_ud_t.hh"


#include <memory>  
# define IDXTYPEWIDTH 64
# define REALTYPEWIDTH 32
extern "C" {
  #include <metis.h>
}
#include <chrono>
#include <atomic>


//TODO: consider restructuring the meta-hnsw
//TODO: update/insert/delete
using dhnsw_idx_t = int64_t;
typedef int storage_idx_t;

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using dhnsw::DhnswService;
using dhnsw::Empty;
using dhnsw::MetaHnswResponse;
using dhnsw::MappingResponse;
using dhnsw::MappingEntry;

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using dhnsw::DhnswService;
using dhnsw::Empty;
using dhnsw::MetaHnswResponse;
using dhnsw::MappingResponse;
using dhnsw::MappingEntry;
using dhnsw::OverflowResponse;
using dhnsw::Offset_SubHnswResponse;
using dhnsw::Offset_ParaResponse;

/*DistributedHnsw method*/

using namespace std::chrono;

// Method to build
DistributedHnsw::DistributedHnsw()
    : d(0),
      num_sub_hnsw(0),
      meta_M(0),
      sub_M(0),
      meta_hnsw(nullptr),
      num_meta(0) {
    // Do not allocate or initialize any resources
}

DistributedHnsw::DistributedHnsw(const DistributedHnsw& other) {
    d = other.d;
    num_sub_hnsw = other.num_sub_hnsw;
    meta_M = other.meta_M;
    sub_M = other.sub_M;
    num_meta = other.num_meta;
    
    // Deep copy meta_hnsw
    if (other.meta_hnsw) {
        meta_hnsw = new faiss::IndexHNSWFlat(*other.meta_hnsw);
    } else {
        meta_hnsw = nullptr;
    }
    
    // Deep copy sub_hnsw vector
    sub_hnsw.resize(other.sub_hnsw.size());
    for (size_t i = 0; i < other.sub_hnsw.size(); i++) {
        if (other.sub_hnsw[i]) {
            sub_hnsw[i] = new faiss::IndexHNSWFlat(*other.sub_hnsw[i]);
        } else {
            sub_hnsw[i] = nullptr;
        }
    }
    
    // Copy other members
    mapping = other.mapping;
}

static std::vector<size_t>
kmeanspp_sample(const float* xb, size_t N, int d, size_t k, std::mt19937& rng)
{
    std::uniform_int_distribution<size_t> uni(0, N-1);
    std::vector<size_t> centers; centers.reserve(k);
    centers.push_back( uni(rng) );                 // 1st center random

    std::vector<float> dist2(N, std::numeric_limits<float>::max());

    for(size_t it=1; it<k; ++it){
        size_t last = centers.back();
        const float* c = xb + last*d;
        for(size_t i=0;i<N;++i){
            float d2 = faiss::fvec_L2sqr(c, xb+i*d, d);
            if (d2 < dist2[i]) dist2[i] = d2;
        }
        // prob ∝ dist2
        std::discrete_distribution<size_t> disc(dist2.begin(), dist2.end());
        centers.push_back(disc(rng));
    }
    return centers;
}


static std::vector<size_t>
approx_kmeanspp_sample(const float* xb, size_t N, int d, size_t k, std::mt19937& rng, size_t sample_per_iter = 100)
{
    // Guard: Log and clamp k to N when N < k (e.g., small buffered_vectors during reconstruction)
    if (N < k) {
        std::cerr << "[approx_kmeanspp_sample] WARNING: N=" << N << " < k=" << k
                  << ". Clamping k to N to avoid assertion failure." << std::endl;
        k = N;
    }

    // Handle edge case: if N == 0, return empty centers
    if (N == 0) {
        std::cerr << "[approx_kmeanspp_sample] WARNING: N=0, returning empty centers." << std::endl;
        return {};
    }

    // Handle edge case: if k == 0, return empty centers
    if (k == 0) {
        return {};
    }

    std::uniform_int_distribution<size_t> uni(0, N - 1);
    std::vector<size_t> centers;
    centers.reserve(k);
    centers.push_back(uni(rng));
    while (centers.size() < k) {
        // Adjust sample_per_iter to not exceed remaining unique candidates
        size_t remaining = N - centers.size();
        size_t actual_sample_size = std::min(sample_per_iter, remaining);
        if (actual_sample_size == 0) {
            // All points are already centers
            break;
        }

        std::vector<size_t> candidate_pool;
        std::unordered_set<size_t> seen(centers.begin(), centers.end());  // Exclude existing centers

        // Limit attempts to avoid infinite loop when N is small
        size_t max_attempts = actual_sample_size * 10;
        size_t attempts = 0;
        while (candidate_pool.size() < actual_sample_size && attempts < max_attempts) {
            size_t candidate = uni(rng);
            if (seen.insert(candidate).second) {
                candidate_pool.push_back(candidate);
            }
            attempts++;
        }

        // If we couldn't find any new candidates, break
        if (candidate_pool.empty()) {
            break;
        }

        size_t best_candidate = candidate_pool[0];
        float max_dist = -1.0f;

        for (size_t id : candidate_pool) {
            const float* x = xb + id * d;
            float min_dist = std::numeric_limits<float>::max();

            for (size_t c : centers) {
                float dist = faiss::fvec_L2sqr(x, xb + c * d, d);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }

            if (min_dist > max_dist) {
                max_dist = min_dist;
                best_candidate = id;
            }
        }

        centers.push_back(best_candidate);
    }

    return centers;
}

void capacity_constrained_kmeans(const float* data, size_t N, int d,
                                 size_t K, size_t L, size_t max_iters,
                                 std::vector<int>& assignment) {
    // Handle edge case: if N == 0, assign nothing
    if (N == 0) {
        std::cerr << "[capacity_constrained_kmeans] WARNING: N=0, returning empty assignment." << std::endl;
        assignment.clear();
        return;
    }

    // Handle edge case: if N < K, reduce K to N (each point is its own cluster)
    size_t effective_K = std::min(K, N);
    if (effective_K < K) {
        std::cerr << "[capacity_constrained_kmeans] WARNING: N=" << N << " < K=" << K
                  << ". Reducing effective K to " << effective_K << "." << std::endl;
    }

    size_t cap = (N + effective_K - 1) / effective_K;
    std::mt19937 rng(17);

    std::vector<std::vector<float>> centroids(effective_K, std::vector<float>(d));
    auto init_ids = approx_kmeanspp_sample(data, N, d, effective_K, rng);

    // Guard: handle case where init_ids has fewer elements than expected
    size_t num_centroids = std::min(init_ids.size(), effective_K);
    if (num_centroids == 0) {
        std::cerr << "[capacity_constrained_kmeans] WARNING: No initial centers returned, "
                  << "assigning all points to cluster 0." << std::endl;
        assignment.assign(N, 0);
        return;
    }

    for (size_t i = 0; i < num_centroids; ++i) {
        std::copy(data + init_ids[i]*d,
                  data + (init_ids[i]+1)*d,
                  centroids[i].begin());
    }
    // If we have fewer centroids than effective_K, duplicate the last one
    for (size_t i = num_centroids; i < effective_K; ++i) {
        centroids[i] = centroids[num_centroids - 1];
    }

    assignment.assign(N, -1);
    std::vector<size_t> ptr(N, 0);
    std::vector<std::array<std::pair<float,int>,  /*L*/ 5>> nearest;
    nearest.resize(N);

    for (size_t iter = 0; iter < max_iters; ++iter) {
        for (size_t i = 0; i < N; ++i) {
            std::priority_queue<std::pair<float,int>> pq;
            for (size_t j = 0; j < effective_K; ++j) {
                float dist = faiss::fvec_L2sqr(data + i*d, centroids[j].data(), d);
                if (pq.size() < L || dist < pq.top().first) {
                    if (pq.size() == L) pq.pop();
                    pq.push({dist, (int)j});
                }
            }
            for (int t = L-1; t >= 0; --t) {
                nearest[i][t] = pq.top(); pq.pop();
            }
            ptr[i] = 0;
        }


        using E = std::tuple<float,size_t,int>;
        std::priority_queue<E, std::vector<E>, std::greater<E>> heap;
        std::vector<int> size(effective_K, 0);
        int assigned = 0;
        assignment.assign(N, -1);

        for (size_t i = 0; i < N; ++i) {
            auto [d0, c0] = nearest[i][0];
            heap.push({d0, i, c0});
        }

        while (assigned < (int)N) {
            auto [dist, pid, cid] = heap.top(); heap.pop();
            if (assignment[pid] != -1) continue;
            if (size[cid] < (int)cap) {
                assignment[pid] = cid;
                size[cid]++;
                assigned++;
            } else {
                ptr[pid]++;
                if (ptr[pid] < (int)L) {
                    auto [d1,c1] = nearest[pid][ ptr[pid] ];
                    heap.push({d1, pid, c1});
                } else {
                    for (int j = 0; j < (int)effective_K; ++j) {
                        if (size[j] < (int)cap) {
                            float d2 = faiss::fvec_L2sqr(data + static_cast<size_t>(pid)*d, centroids[j].data(), d);
                            heap.push({d2, pid, j});
                            break;
                        }
                    }
                }
            }
        }

        std::vector<std::vector<float>> newc(effective_K, std::vector<float>(d, 0));
        std::vector<int> counts(effective_K, 0);
        for (size_t i = 0; i < N; ++i) {
            int c = assignment[i];
            auto& ctr = newc[c];
            for (int z = 0; z < d; ++z) ctr[z] += data[i*d + z];
            counts[c]++;
        }
        for (size_t j = 0; j < effective_K; ++j) {
            if (counts[j] > 0) {
                float inv = 1.0f / counts[j];
                for (int z = 0; z < d; ++z) newc[j][z] *= inv;
            }
        }
        centroids.swap(newc);
    }

}

void DistributedHnsw::build(const std::vector<float>& data_vec, size_t /*unused*/) {
    size_t N = data_vec.size() / d;
    const float* data = data_vec.data();
    size_t K = num_sub_hnsw;   
    size_t L = 5;              
    size_t max_iters = 15;

    std::vector<int> assignment;
    capacity_constrained_kmeans(data, N, d, K, L, max_iters, assignment);

    std::vector<std::vector<dhnsw_idx_t>> buckets(K);
    for (size_t i = 0; i < N; ++i) {
        buckets[ assignment[i] ].push_back(i);
    }

    sub_hnsw.clear(); mapping.clear();
    sub_hnsw.reserve(K); mapping.reserve(K);
    for (size_t j = 0; j < K; ++j) {
        auto& idxs = buckets[j];
        size_t sz = idxs.size();
        if (sz == 0) continue;
        std::vector<float> buf(sz * d);
        for (size_t t = 0; t < sz; ++t) {
            std::copy(data + idxs[t]*d,
                      data + (idxs[t]+1)*d,
                      buf.begin() + t*d);
        }
        auto* index = new faiss::IndexHNSWFlat(d, sub_M);
        index->hnsw.efConstruction = 120;
        index->add(sz, buf.data());
        sub_hnsw.push_back(index);
        mapping.push_back(idxs);
    }

    if (meta_hnsw) delete meta_hnsw;
    meta_hnsw = new faiss::IndexHNSWFlat(d, meta_M);
    for (size_t j = 0; j < sub_hnsw.size(); ++j) {
        auto& idxs = mapping[j];
        std::vector<float> ctr(d, 0);
        for (auto pid : idxs) {
            for (int z = 0; z < d; ++z) ctr[z] += data[pid*d+z];
        }
        float inv = 1.0f / idxs.size();
        for (int z = 0; z < d; ++z) ctr[z] *= inv;
        meta_hnsw->add(1, ctr.data());
    }
    // print the size of each sub_hnsw
    for (int i = 0; i < num_sub_hnsw; i++) {
        printf("Sub-hnsw %d has %ld elements\n", i, sub_hnsw[i]->ntotal);
    }
}

void DistributedHnsw::print_subhnsw_balance() const {
    std::vector<size_t> sizes;
    for (const auto* index : sub_hnsw) {
        sizes.push_back(index ? index->ntotal : 0);
    }

    size_t sum = std::accumulate(sizes.begin(), sizes.end(), size_t(0));
    double mean = static_cast<double>(sum) / sizes.size();

    double sq_sum = 0.0;
    for (auto s : sizes) {
        double diff = s - mean;
        sq_sum += diff * diff;
    }
    double stddev = std::sqrt(sq_sum / sizes.size());

    size_t min_val = *std::min_element(sizes.begin(), sizes.end());
    size_t max_val = *std::max_element(sizes.begin(), sizes.end());

    std::cout << "====== Sub-HNSW Balance Report ======\n";
    std::cout << "Num sub_hnsw: " << sizes.size() << "\n";
    std::cout << "Total points: " << sum << "\n";
    std::cout << "Mean size: " << mean << "\n";
    std::cout << "Stddev size: " << stddev << "\n";
    std::cout << "Min size: " << min_val << "\n";
    std::cout << "Max size: " << max_val << "\n";
}

 size_t DistributedHnsw::get_meta_hnsw_size() const {
            // Calculate the total memory size of meta_hnsw in MB
            
            // Size of the index structure itself
            size_t base_size = sizeof(faiss::IndexHNSWFlat);
            
            // Size of vector data
            size_t xb_size = 0;
            if (meta_hnsw->storage) {
                xb_size = meta_hnsw->ntotal * meta_hnsw->d * sizeof(float);
            }
            
            // Size of HNSW graph structures
            size_t levels_size = meta_hnsw->hnsw.levels.capacity() * sizeof(int);
            size_t offsets_size = meta_hnsw->hnsw.offsets.capacity() * sizeof(size_t);
            size_t neighbors_size = meta_hnsw->hnsw.neighbors.capacity() * sizeof(int);
            
            // Total size in bytes
            size_t total_bytes = base_size + xb_size + levels_size + offsets_size + neighbors_size;
            

            return total_bytes;
}

 size_t DistributedHnsw::get_per_sub_hnsw_size() const {
            // Calculate the total memory size of meta_hnsw in MB
            
            // Size of the index structure itself
            size_t base_size = sizeof(faiss::IndexHNSWFlat);
            
            // Size of vector data
            size_t xb_size = 0;
            if (sub_hnsw[0]->storage) {
                xb_size = sub_hnsw[0]->ntotal * meta_hnsw->d * sizeof(float);
            }
            
            // Size of HNSW graph structures
            size_t levels_size = sub_hnsw[0]->hnsw.levels.capacity() * sizeof(int);
            size_t offsets_size = sub_hnsw[0]->hnsw.offsets.capacity() * sizeof(size_t);
            size_t neighbors_size = sub_hnsw[0]->hnsw.neighbors.capacity() * sizeof(int);
            
            // Total size in bytes
            size_t total_bytes = base_size + xb_size + levels_size + offsets_size + neighbors_size;
            

            return total_bytes;
}

// Method to search meta-hnsw,result record in sub_hnsw_tosearch
void DistributedHnsw::meta_search(const int n,const float* query, int K_meta, float* distances, dhnsw_idx_t* labels, std::vector<int>& sub_hnsw_tosearch) {
    meta_hnsw->search(n, query, K_meta, distances, labels);
    for(int i = 0; i < K_meta * n ; i++) {
        int label = labels[i];
        sub_hnsw_tosearch.push_back(label);
    }
}

// Method to search sub-hnsw, result record in distances and labels
void DistributedHnsw::sub_search(const int n, const float* query, int K_meta, int K_sub, float* distances, dhnsw_idx_t* labels, std::vector<int>& sub_hnsw_tosearch, dhnsw_idx_t* sub_hnsw_tags) {
    std::vector<std::tuple<float, dhnsw_idx_t, dhnsw_idx_t>> result;
    std::vector<std::unordered_set<int>> searchset(num_sub_hnsw);// vcector:sub_hnsw unordered_set:query index

    for (int i = 0; i < n * K_meta; i++) {
        int query_idx = i / K_meta;
        int sub_idx = sub_hnsw_tosearch[i];
        searchset[sub_idx].insert(query_idx);
    }


    for (int i = 0; i < num_sub_hnsw; i++) {
        size_t num_queries = searchset[i].size();
        if (num_queries > 0) {

            float* tmp_sub_distances = new float[K_sub * num_queries];
            dhnsw_idx_t* tmp_sub_labels = new dhnsw_idx_t[K_sub * num_queries];
            float* tmp_query = new float[d * num_queries];

            int j = 0;
            std::vector<int> query_indices(num_queries);
            for (auto it = searchset[i].begin(); it != searchset[i].end(); ++it, ++j) {
                int tmp_query_idx = *it;
                query_indices[j] = tmp_query_idx;
                std::copy(query + tmp_query_idx * d, query + (tmp_query_idx + 1) * d, tmp_query + j * d);
            }

            sub_hnsw[i]->search(num_queries, tmp_query, K_sub, tmp_sub_distances, tmp_sub_labels);

            // merge the results
            for (size_t j = 0; j < num_queries; j++) {
                int tmp_query_idx = query_indices[j];
                result.clear();

                for (int k = 0; k < K_sub; k++) {
                    if (labels[tmp_query_idx * K_sub + k] != -1) {
                        result.emplace_back(
                            distances[tmp_query_idx * K_sub + k],
                            labels[tmp_query_idx * K_sub + k],
                            sub_hnsw_tags[tmp_query_idx * K_sub + k]);
                    }
                }

                for (int k = 0; k < K_sub; k++) {
                    result.emplace_back(
                        tmp_sub_distances[j * K_sub + k],
                        tmp_sub_labels[j * K_sub + k],
                        i);
                }


                std::sort(result.begin(), result.end());

                for (int k = 0; k < K_sub && k < result.size(); k++) {
                    distances[tmp_query_idx * K_sub + k] = std::get<0>(result[k]);
                    labels[tmp_query_idx * K_sub + k] = std::get<1>(result[k]);
                    sub_hnsw_tags[tmp_query_idx * K_sub + k] = std::get<2>(result[k]);
                }
            }

            delete[] tmp_sub_distances;
            delete[] tmp_sub_labels;
            delete[] tmp_query;
        }
    }
}

// Method to search using the hierarchical index, result record in distances and labels (for single machine)
void DistributedHnsw::hierarchicalSearch(const int n,const float* query, int K_meta, int K_sub, float* distances, dhnsw_idx_t* labels, dhnsw_idx_t* sub_hnsw_tags, dhnsw_idx_t* original_index,int efSearch) {
    //initialize the distances and labels and sub_hnsw_tags
    std::fill(distances, distances + n * K_sub, std::numeric_limits<float>::max());
    std::fill(labels, labels + n * K_sub, -1); 
    std::fill(sub_hnsw_tags, sub_hnsw_tags + n * K_sub, -1);
    meta_hnsw->hnsw.efSearch = efSearch;
    for(int i = 0; i < num_sub_hnsw; i++) {
        sub_hnsw[i]->hnsw.efSearch = efSearch;
    }
    float* meta_distances = new float[K_meta * n];
    dhnsw_idx_t* meta_labels = new dhnsw_idx_t[K_meta* n];
    std::vector<int> sub_hnsw_tosearch;
    meta_search(n, query, K_meta, meta_distances, meta_labels, sub_hnsw_tosearch);
    delete[] meta_distances;
    delete[] meta_labels;
    sub_search(n, query, K_meta, K_sub, distances, labels, sub_hnsw_tosearch, sub_hnsw_tags);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < K_sub; j++) {
            original_index[i * K_sub + j] = mapping[sub_hnsw_tags[i * K_sub + j]][labels[i * K_sub + j]];
        }
    }
}


// Method to insert
//TODO: error handling
void DistributedHnsw::insert(const int n, const std::vector<float>& data) {
    // std::cout<<n<<std::endl;
    float* construct_distances = new float[n];
    dhnsw_idx_t* construct_labels = new dhnsw_idx_t[n];
    std::vector<int> sub_hnsw_toinsert; 
    this->meta_search(n, data.data(), 1, construct_distances, construct_labels, sub_hnsw_toinsert);
    delete[] construct_distances;
    delete[] construct_labels;
    std::vector<std::vector<float>> insertset(num_sub_hnsw);// vcector:sub_hnsw unordered_set:query index 
    for (int i = 0; i < n ; i++) {
        int sub_idx = sub_hnsw_toinsert[i];
        if(sub_idx >= num_sub_hnsw){
            std::cerr << "Invalid sub-index: " << sub_idx << std::endl;
            continue;
        }
        insertset[sub_idx].insert(insertset[sub_idx].end(), data.begin() + i * d, data.begin() + (i + 1) * d);
        mapping[sub_idx].push_back(i);
    } 
    for (int i = 0; i < num_sub_hnsw; i++) {
        size_t num_inserts = insertset[i].size();
        if (!insertset[i].empty()) {
            sub_hnsw[i]->add(insertset[i].size() / d, insertset[i].data());
        }
    }
}


std::vector<uint8_t> DistributedHnsw::serialize() const {
    std::vector<uint8_t> buffer;
    std::ostringstream oss(std::ios::binary);

    // 1. Serialize primitive data types
    oss.write(reinterpret_cast<const char*>(&d), sizeof(d));
    oss.write(reinterpret_cast<const char*>(&num_sub_hnsw), sizeof(num_sub_hnsw));
    oss.write(reinterpret_cast<const char*>(&meta_M), sizeof(meta_M));
    oss.write(reinterpret_cast<const char*>(&sub_M), sizeof(sub_M));
    oss.write(reinterpret_cast<const char*>(&num_meta), sizeof(num_meta));

    // 2. Serialize the meta_hnsw index
    {
        faiss::VectorIOWriter writer;
        faiss::write_index(meta_hnsw, &writer);
        // Write the size of the serialized index
        size_t meta_index_size = writer.data.size();
        oss.write(reinterpret_cast<const char*>(&meta_index_size), sizeof(meta_index_size));
        // Write the serialized index data
        oss.write(reinterpret_cast<const char*>(writer.data.data()), meta_index_size);
    }

    // 3. Serialize the sub_hnsw indices
    {
        // Write the number of sub_hnsw indices
        size_t num_sub_indices = sub_hnsw.size();
        oss.write(reinterpret_cast<const char*>(&num_sub_indices), sizeof(num_sub_indices));

        for (const auto& index : sub_hnsw) {
            faiss::VectorIOWriter writer;
            faiss::write_index(index, &writer);
            // Write the size of the serialized index
            size_t index_size = writer.data.size();
            oss.write(reinterpret_cast<const char*>(&index_size), sizeof(index_size));
            // Write the serialized index data
            oss.write(reinterpret_cast<const char*>(writer.data.data()), index_size);
        }
    }


    // 5. Serialize the 'mapping' vector of vectors
    {
        size_t mapping_size = mapping.size();
        oss.write(reinterpret_cast<const char*>(&mapping_size), sizeof(mapping_size));

        for (const auto& vec : mapping) {
            size_t vec_size = vec.size();
            oss.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
            oss.write(reinterpret_cast<const char*>(vec.data()), vec_size * sizeof(dhnsw_idx_t));
        }
    }


    // Convert the ostringstream to a vector<uint8_t>
    std::string str = oss.str();
    buffer.assign(str.begin(), str.end());

    return buffer;
}

DistributedHnsw DistributedHnsw::deserialize(const std::vector<uint8_t>& data) {
    DistributedHnsw obj;  // Temporary initialization
    std::istringstream iss(std::string(data.begin(), data.end()), std::ios::binary);

    // 1. Deserialize primitive data types
    iss.read(reinterpret_cast<char*>(&obj.d), sizeof(obj.d));
    iss.read(reinterpret_cast<char*>(&obj.num_sub_hnsw), sizeof(obj.num_sub_hnsw));
    iss.read(reinterpret_cast<char*>(&obj.meta_M), sizeof(obj.meta_M));
    iss.read(reinterpret_cast<char*>(&obj.sub_M), sizeof(obj.sub_M));
    iss.read(reinterpret_cast<char*>(&obj.num_meta), sizeof(obj.num_meta));

    obj.mapping.resize(obj.num_sub_hnsw);
    obj.sub_hnsw.resize(obj.num_sub_hnsw, nullptr);

    // 2. Deserialize the meta_hnsw index
    {
        size_t meta_index_size;
        iss.read(reinterpret_cast<char*>(&meta_index_size), sizeof(meta_index_size));

        std::vector<uint8_t> index_data(meta_index_size);
        iss.read(reinterpret_cast<char*>(index_data.data()), meta_index_size);

        faiss::VectorIOReader reader;
        reader.data.assign(index_data.begin(), index_data.end());

        obj.meta_hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(&reader));

        assert(obj.meta_hnsw != nullptr && "Failed to deserialize meta_hnsw index");
    }

    // 3. Deserialize the sub_hnsw indices
    {
        size_t num_sub_indices;
        iss.read(reinterpret_cast<char*>(&num_sub_indices), sizeof(num_sub_indices));
        obj.sub_hnsw.resize(num_sub_indices, nullptr);

        for (size_t i = 0; i < num_sub_indices; ++i) {
            size_t index_size;
            iss.read(reinterpret_cast<char*>(&index_size), sizeof(index_size));

            std::vector<uint8_t> index_data(index_size);
            iss.read(reinterpret_cast<char*>(index_data.data()), index_size);

            faiss::VectorIOReader reader;
            reader.data.assign(index_data.begin(), index_data.end());

            obj.sub_hnsw[i] = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(&reader));

            assert(obj.sub_hnsw[i] != nullptr && "Failed to deserialize sub_hnsw index");
        }
    }

    // 5. Deserialize the 'mapping' vector of vectors
    {
        size_t mapping_size;
        iss.read(reinterpret_cast<char*>(&mapping_size), sizeof(mapping_size));
        obj.mapping.resize(mapping_size);

        for (size_t i = 0; i < mapping_size; ++i) {
            size_t vec_size;
            iss.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));
            obj.mapping[i].resize(vec_size);
            iss.read(reinterpret_cast<char*>(obj.mapping[i].data()), vec_size * sizeof(dhnsw_idx_t));
        }
    }

    return obj;
}

void DistributedHnsw::insert_with_record(const int n, const std::vector<float>& data) {
    // std::cout<<n<<std::endl;
    float* construct_distances = new float[n];
    dhnsw_idx_t* construct_labels = new dhnsw_idx_t[n];
    std::vector<int> sub_hnsw_toinsert; 
    this->meta_search(n, data.data(), 1, construct_distances, construct_labels, sub_hnsw_toinsert);
    delete[] construct_distances;
    delete[] construct_labels;
    std::vector<std::vector<float>> insertset(num_sub_hnsw);// vcector:sub_hnsw unordered_set:query index 
    for (int i = 0; i < n ; i++) {
        int sub_idx = sub_hnsw_toinsert[i];
        if(sub_idx >= num_sub_hnsw){
            std::cerr << "Invalid sub-index: " << sub_idx << std::endl;
            continue;
        }
        insertset[sub_idx].insert(insertset[sub_idx].end(), data.begin() + i * d, data.begin() + (i + 1) * d);
        mapping[sub_idx].push_back(i);
    } 
    for (int i = 0; i < num_sub_hnsw; i++) {
        size_t num_inserts = insertset[i].size();
        if (!insertset[i].empty()) {
            sub_hnsw[i]->add(insertset[i].size() / d, insertset[i].data());
        }
    }
}

std::vector<uint8_t> DistributedHnsw::serialize_with_record(std::vector<uint64_t>& offset) const {
    std::vector<uint8_t> buffer;
    std::ostringstream oss(std::ios::binary);

    // The 'offset' vector will store 2 entries per sub_hnsw:
    // offset[i * 2 + 0] = sub_hnsw[i] begin position
    // offset[i * 2 + 1] = sub_hnsw[i] end position
    offset.resize(2 * num_sub_hnsw);

    // Serialize primitive data types
    oss.write(reinterpret_cast<const char*>(&d), sizeof(d));
    oss.write(reinterpret_cast<const char*>(&num_sub_hnsw), sizeof(num_sub_hnsw));
    oss.write(reinterpret_cast<const char*>(&meta_M), sizeof(meta_M));
    oss.write(reinterpret_cast<const char*>(&sub_M), sizeof(sub_M));
    oss.write(reinterpret_cast<const char*>(&num_meta), sizeof(num_meta));

    // Serialize the meta_hnsw index
    {
        faiss::VectorIOWriter writer;
        faiss::write_index(meta_hnsw, &writer);

        size_t meta_index_size = writer.data.size();
        oss.write(reinterpret_cast<const char*>(&meta_index_size), sizeof(meta_index_size));
        oss.write(reinterpret_cast<const char*>(writer.data.data()), meta_index_size);
    }

    // Serialize the sub_hnsw indices
    {
        size_t num_sub_indices = sub_hnsw.size();
        oss.write(reinterpret_cast<const char*>(&num_sub_indices), sizeof(num_sub_indices));

        for (size_t i = 0; i < sub_hnsw.size(); ++i) {
            offset[i * 2 + 0] = oss.tellp();

            faiss::VectorIOWriter writer;
            faiss::write_index(sub_hnsw[i], &writer);

            size_t index_size = writer.data.size();
            oss.write(reinterpret_cast<const char*>(&index_size), sizeof(index_size));
            oss.write(reinterpret_cast<const char*>(writer.data.data()), index_size);

            offset[i * 2 + 1] = oss.tellp();
        }
    }


    // Serialize the 'mapping' vector of vectors
    {
        size_t mapping_size = mapping.size();
        oss.write(reinterpret_cast<const char*>(&mapping_size), sizeof(mapping_size));

        for (const auto& vec : mapping) {
            size_t vec_size = vec.size();
            oss.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
            oss.write(reinterpret_cast<const char*>(vec.data()), vec_size * sizeof(dhnsw_idx_t));
        }
    }

    // Convert the ostringstream to a vector<uint8_t>
    std::string str = oss.str();
    buffer.assign(str.begin(), str.end());

    return buffer;
}

std::vector<uint8_t> DistributedHnsw::serialize_with_record_with_gap(std::vector<uint64_t>& offset) const {
    std::vector<uint8_t> buffer;
    std::ostringstream oss(std::ios::binary);

    // The 'offset' vector will store 5 entries per sub_hnsw:
    // offset[i * 2 + 0] = sub_hnsw[i] begin position
    // offset[i * 2 + 1] = sub_hnsw[i] end position 
    offset.resize(2 * num_sub_hnsw); 

    // 1. Serialize primitive data types
    oss.write(reinterpret_cast<const char*>(&d), sizeof(d));
    oss.write(reinterpret_cast<const char*>(&num_sub_hnsw), sizeof(num_sub_hnsw));
    oss.write(reinterpret_cast<const char*>(&meta_M), sizeof(meta_M));
    oss.write(reinterpret_cast<const char*>(&sub_M), sizeof(sub_M));
    oss.write(reinterpret_cast<const char*>(&num_meta), sizeof(num_meta));

    // 2. Serialize the meta_hnsw index
    size_t meta_index_size = 0; 
    {
        faiss::VectorIOWriter writer;
        faiss::write_index(meta_hnsw, &writer);

        // Write the size of the serialized index
        meta_index_size = writer.data.size();
        oss.write(reinterpret_cast<const char*>(&meta_index_size), sizeof(meta_index_size));

        // Write the serialized index data
        oss.write(reinterpret_cast<const char*>(writer.data.data()), meta_index_size);
    }

        // 3. Serialize the sub_hnsw index and data with gap and record offsets
    {
        // Write the number of sub_hnsw indices
        size_t num_sub_indices = sub_hnsw.size();
        oss.write(reinterpret_cast<const char*>(&num_sub_indices), sizeof(num_sub_indices));
        std::vector<char> index_gap(meta_index_size, 0);
        for (size_t i = 0; i < sub_hnsw.size(); ++i) {

            offset[i * 2]=oss.tellp(); // This is the start position of sub_hnsw[i] index
            // Serialize the sub_hnsw index
            faiss::VectorIOWriter writer;
            faiss::write_index(sub_hnsw[i], &writer);
            
            // // Write the size of the serialized index
            size_t index_size = writer.data.size();
            // oss.write(reinterpret_cast<const char*>(&index_size), sizeof(index_size));
            
            
            // Write the serialized index data
            oss.write(reinterpret_cast<const char*>(writer.data.data()), index_size);

            offset[i * 2 + 1] = oss.tellp(); // This is the end position of sub_hnsw[i] index

            // Add a gap equal to the size of the serialized index
            oss.write(index_gap.data(), meta_index_size);

        }
    }

    // 5. Serialize the 'mapping' vector of vectors
    {
        size_t mapping_size = mapping.size();
        oss.write(reinterpret_cast<const char*>(&mapping_size), sizeof(mapping_size));

        for (const auto& vec : mapping) {
            size_t vec_size = vec.size();
            oss.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
            oss.write(reinterpret_cast<const char*>(vec.data()), vec_size * sizeof(dhnsw_idx_t));
        }
    }

    // Convert the ostringstream to a vector<uint8_t>
    std::string str = oss.str();
    buffer.assign(str.begin(), str.end());

    return buffer;
}

std::vector<uint8_t> DistributedHnsw::serialize_with_record_with_in_out_gap(std::vector<uint64_t>& offset_sub_hnsw, std::vector<uint64_t>& offset_para, std::vector<uint64_t>& overflow) const {
      for(int i = 0; i < num_sub_hnsw; i++) {
        // 2 entries for each sub_hnsw
        offset_sub_hnsw.push_back(0); // start
        offset_sub_hnsw.push_back(0); // end without gap

        // 9 entries for each sub_hnsw
        offset_para.push_back(0); // idx->ntotal (idx_t) 0
        offset_para.push_back(0); // hnsw->levels (start,change size from here) 1  , calculate end from here
        offset_para.push_back(0); // hnsw->offsets (start,change size from here) 2 , calculate end from here, max boundary of levels
        offset_para.push_back(0); // hnsw->neighbors (start,change size from here) 3 , calculate end from here, max boundary of offsets
        offset_para.push_back(0); // hnsw->entry_point (storage_idx_t) 4, max boundary of neighbors
        offset_para.push_back(0); // hnsw->max_level (int) 5, (can optimized as entry_point+1)
        offset_para.push_back(0); // storage->header->ntotal (idx_t) 6
        offset_para.push_back(0); // storage->idxf->xb (start, change size from here) 7  && get end calculate from here
        offset_para.push_back(0); // storage->idxf->xb  max boundary 8 update in local only



        
        overflow.push_back(0); // overflow gap start 0
        overflow.push_back(0); // overflow gap end 1
        overflow.push_back(0); // overflow gap max boundary 2
    }
        
    faiss::VectorIOWriter writer;
    size_t current_pos = 0;

    // Write header information
    writer(reinterpret_cast<const char*>(&d), sizeof(d), 1);
    current_pos += sizeof(d);
    writer(reinterpret_cast<const char*>(&num_sub_hnsw), sizeof(num_sub_hnsw), 1);
    current_pos += sizeof(num_sub_hnsw);
    writer(reinterpret_cast<const char*>(&meta_M), sizeof(meta_M), 1);
    current_pos += sizeof(meta_M);
    writer(reinterpret_cast<const char*>(&sub_M), sizeof(sub_M), 1);
    current_pos += sizeof(sub_M);
    writer(reinterpret_cast<const char*>(&num_meta), sizeof(num_meta), 1);
    current_pos += sizeof(num_meta);

    // Serialize meta HNSW index to get its size for overflow gaps
    size_t meta_start = current_pos;
    faiss::write_index(meta_hnsw, &writer);
    size_t meta_size = writer.data.size() - meta_start;
    current_pos += meta_size;
    std::cout << "Meta HNSW size: " << meta_size << std::endl;
    // Write number of sub indices
    size_t num_sub_indices = sub_hnsw.size();
    writer(reinterpret_cast<const char*>(&num_sub_indices), sizeof(num_sub_indices), 1);
    current_pos += sizeof(num_sub_indices);

    // Serialize each sub HNSW with overflow gaps between pairs
    for (size_t i = 0; i < sub_hnsw.size(); ++i) {
        offset_sub_hnsw[i * 2] = current_pos;
        faiss::write_dhnsw_index_init_(sub_hnsw[i], &writer, offset_para, i, current_pos);
        offset_sub_hnsw[i * 2 + 1] = current_pos;
        // Add overflow gap between every second sub-HNSW 0-1, 2-3, 4-5, 6-7
        if(i % 2 == 0 ){
            size_t gap_size = meta_size; // Gap size is meta HNSW size
            size_t gap_start = current_pos;
            size_t gap_end = current_pos + gap_size;
            // overflow for i
            overflow[i * 3] = gap_start;
            overflow[i * 3 + 1] = gap_start; // no overflow now
            overflow[i * 3 + 2] = gap_end; // max boundary 2
            
            // overflow for i+1
            if(i != sub_hnsw.size() - 1){
                overflow[(i+1) * 3] = gap_end;
                overflow[(i+1) * 3 + 1] = gap_end; // no overflow now
                overflow[(i+1) * 3 + 2] = gap_start; // max boundary 2 !!! important
            }
            std::vector<char> gap(gap_size, 0);
            writer(gap.data(), 1, gap_size);
            current_pos += gap_size;
        }
        
    }

    return writer.data; 
}

DistributedHnsw DistributedHnsw::deserialize_with_record_with_in_out_gap(
    const std::vector<uint8_t>& data,
    std::vector<uint64_t>& offset_sub_hnsw, 
    std::vector<uint64_t>& offset_para, 
    std::vector<uint64_t>& overflow
) const{
    DistributedHnsw obj;
    faiss::VectorIOReader reader;
    reader.data.assign(data.begin(), data.end());
    size_t current_pos = 0;

    // Read header information
    std::cout << "Reading header..." << std::endl;
    reader(reinterpret_cast<char*>(&obj.d), sizeof(obj.d), 1); 
    current_pos += sizeof(obj.d);
    std::cout << "Read d: " << obj.d << std::endl;
    
    reader(reinterpret_cast<char*>(&obj.num_sub_hnsw), sizeof(obj.num_sub_hnsw), 1);
    current_pos += sizeof(obj.num_sub_hnsw);
    std::cout << "Read num_sub_hnsw: " << obj.num_sub_hnsw << std::endl;

    reader(reinterpret_cast<char*>(&obj.meta_M), sizeof(obj.meta_M), 1);
    current_pos += sizeof(obj.meta_M);
    std::cout << "Read meta_M: " << obj.meta_M << std::endl;
    
    reader(reinterpret_cast<char*>(&obj.sub_M), sizeof(obj.sub_M), 1);
    current_pos += sizeof(obj.sub_M);
    std::cout << "Read sub_M: " << obj.sub_M << std::endl;

    reader(reinterpret_cast<char*>(&obj.num_meta), sizeof(obj.num_meta), 1);
    current_pos += sizeof(obj.num_meta);
    std::cout << "Read num_meta: " << obj.num_meta << std::endl;

    // Read meta HNSW
    std::cout << "Reading meta_hnsw..." << std::endl;
    size_t meta_start = current_pos;
    obj.meta_hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(&reader));
    if (!obj.meta_hnsw) {
        throw std::runtime_error("Failed to read meta HNSW index");
    }

    size_t meta_size = reader.rp - meta_start;
    current_pos += meta_size;
    std::cout << "Meta HNSW read successfully, size: " << meta_size << std::endl;


    // Read number of sub indices
    size_t num_sub_indices;
    reader(&num_sub_indices, sizeof(num_sub_indices), 1);
    current_pos += sizeof(num_sub_indices);
    std::cout << "Number of sub indices: " << num_sub_indices << std::endl;
    obj.sub_hnsw.resize(num_sub_indices); 

    std::cout << "Processing sub_hnsw indices..." << std::endl; 
    for (size_t i = 0; i < num_sub_indices; ++i) {
        size_t index_start = offset_sub_hnsw[i * 2];
        size_t index_end = offset_sub_hnsw[i * 2 + 1];
        size_t index_size = index_end - index_start;
        current_pos = index_start;
        obj.sub_hnsw[i] = new faiss::IndexHNSWFlat();
            if (!obj.sub_hnsw[i]) {
                throw std::runtime_error("Failed to allocate new IndexHNSWFlat for sub_hnsw " + std::to_string(i));
            }
            
            std::cout << "Created new IndexHNSWFlat at " << obj.sub_hnsw[i] << std::endl;
            
            std::cout << "Calling read_dhnsw_index_init..." << std::endl;
        faiss::read_dhnsw_index_init_(
            obj.sub_hnsw[i], 
            &reader,
            const_cast<std::vector<uint64_t>&>(offset_para),
            i,
            current_pos
        );
        std::cout << "sub_hnsw[" << i << "] read successfully" << std::endl;
        if (obj.sub_hnsw[i]->hnsw.entry_point < 0) {
                std::cout << "Warning: sub_hnsw[" << i << "] may not be properly initialized (entry_point: " 
                          << obj.sub_hnsw[i]->hnsw.entry_point << ")" << std::endl;
            } else {
                std::cout << "Verification: sub_hnsw[" << i << "]->hnsw.max_level = " 
                          << obj.sub_hnsw[i]->hnsw.max_level << std::endl;
                std::cout << "Verification: sub_hnsw[" << i << "]->hnsw.efSearch = " 
                          << obj.sub_hnsw[i]->hnsw.efSearch << std::endl;
        }
    }
    //TODO: overflow
    std::cout << "Sub_hnsw indices read successfully" << std::endl;
    return obj;
}

faiss::IndexHNSWFlat* DistributedHnsw::deserialize_sub_hnsw_with_record_with_in_out_gap(
    const std::vector<uint8_t>& data,
    int sub_idx,
    std::vector<uint64_t>& offset_sub_hnsw,
    std::vector<uint64_t>& offset_para, 
    std::vector<uint64_t>& overflow
) const{
        faiss::VectorIOReader reader;
        reader.data.assign(data.begin(), data.end());
        faiss::IndexHNSWFlat* obj = new faiss::IndexHNSWFlat();
        size_t current_pos = 0;

        for(int i = sub_idx * 9; i < sub_idx * 9 + 9; i++){
            offset_para[i] = offset_para[i] - offset_sub_hnsw[sub_idx * 2];
        }
        obj = new faiss::IndexHNSWFlat();
            if (!obj) {
                throw std::runtime_error("Failed to allocate new IndexHNSWFlat for sub_hnsw " + std::to_string(sub_idx));
            }
            
            std::cout << "Created new IndexHNSWFlat at " << obj << std::endl;
            
            std::cout << "Calling read_dhnsw_index_init..." << std::endl;
        faiss::read_dhnsw_index_init_(
            obj, 
            &reader,
            const_cast<std::vector<uint64_t>&>(offset_para),
            sub_idx,
            current_pos
        );
        for(int i = sub_idx * 9; i < sub_idx * 9 + 9; i++){
            offset_para[i] = offset_para[i] + offset_sub_hnsw[sub_idx * 2];
        }
        std::cout << "sub_hnsw[" << sub_idx << "] read successfully" << std::endl;
        if (obj->hnsw.entry_point < 0) {
                std::cout << "Warning: sub_hnsw[" << sub_idx << "] may not be properly initialized (entry_point: " 
                          << obj->hnsw.entry_point << ")" << std::endl;
            } else {
                std::cout << "Verification: sub_hnsw[" << sub_idx << "]->hnsw.max_level = " 
                          << obj->hnsw.max_level << std::endl;
                std::cout << "Verification: sub_hnsw[" << sub_idx << "]->hnsw.efSearch = " 
                          << obj->hnsw.efSearch << std::endl;
        }
        //TODO: overflow
        return obj;
}
void DistributedHnsw::serialize_with_record_with_gap_to_file(
    const std::string& filename,
    std::vector<uint64_t>& offset  
) {
    FILE* f = std::fopen(filename.c_str(), "wb");
    if (!f) { perror("fopen"); std::exit(1); }

    std::fwrite(&d,              sizeof(d),              1, f);
    std::fwrite(&num_sub_hnsw,   sizeof(num_sub_hnsw),   1, f);
    std::fwrite(&meta_M,         sizeof(meta_M),         1, f);
    std::fwrite(&sub_M,          sizeof(sub_M),          1, f);
    std::fwrite(&num_meta,       sizeof(num_meta),       1, f);

    faiss::write_index(meta_hnsw, f);

    offset.resize(2 * num_sub_hnsw);
    const size_t meta_size = std::ftell(f); 
    std::vector<char> gap(meta_size, 0);

    size_t num_sub = sub_hnsw.size();
    for (size_t i = 0; i < num_sub; i++) {
        offset[i*2 + 0] = std::ftell(f);
        faiss::write_index(sub_hnsw[i], f);
        offset[i*2 + 1] = std::ftell(f);

        std::fwrite(gap.data(), 1, gap.size(), f);
    }

    size_t mapping_size = mapping.size();
    std::fwrite(&mapping_size, sizeof(mapping_size), 1, f);
    for (auto& vec : mapping) {
        size_t vsz = vec.size();
        std::fwrite(&vsz, sizeof(vsz), 1, f);
        std::fwrite(vec.data(), sizeof(dhnsw_idx_t), vsz, f);
    }

    std::fclose(f);
}

DistributedHnsw DistributedHnsw::deserialize_with_record_with_gap(//TODO: sub_hnsw de-serialization is wrong
    const std::vector<uint8_t>& data,
    const std::vector<uint64_t>& offset
) {
    DistributedHnsw obj;
    std::istringstream iss(std::string(data.begin(), data.end()), std::ios::binary);

    // 1. Deserialize primitive data types
    iss.read(reinterpret_cast<char*>(&obj.d), sizeof(obj.d));
    iss.read(reinterpret_cast<char*>(&obj.num_sub_hnsw), sizeof(obj.num_sub_hnsw));
    iss.read(reinterpret_cast<char*>(&obj.meta_M), sizeof(obj.meta_M));
    iss.read(reinterpret_cast<char*>(&obj.sub_M), sizeof(obj.sub_M));
    iss.read(reinterpret_cast<char*>(&obj.num_meta), sizeof(obj.num_meta));

    obj.mapping.resize(obj.num_sub_hnsw);
    obj.sub_hnsw.resize(obj.num_sub_hnsw, nullptr);

    // 2. Deserialize the meta_hnsw index
    {
        size_t meta_index_size;
        iss.read(reinterpret_cast<char*>(&meta_index_size), sizeof(meta_index_size));

        std::vector<uint8_t> index_data(meta_index_size);
        iss.read(reinterpret_cast<char*>(index_data.data()), meta_index_size);

        faiss::VectorIOReader reader;
        reader.data.assign(index_data.begin(), index_data.end());

        obj.meta_hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(&reader));

        assert(obj.meta_hnsw != nullptr && "Failed to deserialize meta_hnsw index");
    }

    // 3. Deserialize the sub_hnsw indices using the offset vector
    {
        size_t num_sub_indices;
        iss.read(reinterpret_cast<char*>(&num_sub_indices), sizeof(num_sub_indices));
        obj.sub_hnsw.resize(num_sub_indices, nullptr);

        for (size_t i = 0; i < num_sub_indices; ++i) {
            size_t index_start = offset[i * 2 + 0];
            size_t index_end = offset[i * 2 + 1];
            size_t index_size = index_end - index_start;

            // Extract the serialized sub-index data using the offsets
            std::vector<uint8_t> index_data(index_size);
            std::copy(
                data.begin() + index_start,
                data.begin() + index_end,
                index_data.begin()
            );

            // Deserialize the sub-index
            faiss::VectorIOReader reader;
            reader.data.assign(index_data.begin(), index_data.end());

            obj.sub_hnsw[i] = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(&reader));
            assert(obj.sub_hnsw[i] != nullptr && "Failed to deserialize sub_hnsw index");
        }
    }


    // 5. Deserialize the 'mapping' vector of vectors
    {
        size_t mapping_size;
        iss.read(reinterpret_cast<char*>(&mapping_size), sizeof(mapping_size));
        obj.mapping.resize(mapping_size);

        for (size_t i = 0; i < mapping_size; ++i) {
            size_t vec_size;
            iss.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));
            obj.mapping[i].resize(vec_size);
            iss.read(reinterpret_cast<char*>(obj.mapping[i].data()), vec_size * sizeof(dhnsw_idx_t));
        }
    }

    return obj;
}


std::vector<faiss::IndexHNSWFlat*> LocalHnsw::deserialize_sub_hnsw_batch_with_gap(const std::vector<uint8_t>& data, const std::vector<uint64_t>& offset, std::vector<int> sub_hnsw_tosearch) {
    std::vector<faiss::IndexHNSWFlat*> index = std::vector<faiss::IndexHNSWFlat*>(sub_hnsw_tosearch.size(), nullptr);
    {
        for (size_t idx = 0; idx < sub_hnsw_tosearch.size(); ++idx) {
            int i = sub_hnsw_tosearch[idx];
            size_t index_start = offset[i * 2];
            size_t index_end = offset[i * 2 + 1];
            size_t index_size = index_end - index_start;
            if (index_end < index_start) {
                std::cerr << "Error: index_end (" << index_end << ") < index_start (" << index_start << ")" << std::endl;
                // Handle the error appropriately
                return {};
            }
            // Extract the serialized sub-index data
            std::vector<uint8_t> index_data(index_size);
            std::copy(
                data.begin() + index_start,
                data.begin() + index_end,
                index_data.begin()
            );

            // Deserialize the sub-index
            faiss::VectorIOReader reader;
            reader.data = index_data;

            index[idx] = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(&reader));
            assert(index[idx] != nullptr && "Failed to deserialize sub_hnsw index");
        }


    }
    return index;
}

std::vector<uint8_t> DistributedHnsw::initial_serialize_whole( 
            std::vector<uint64_t>& offset_sub_hnsw,
            std::vector<uint64_t>& offset_para,
            std::vector<uint64_t>& overflow
        ) const {
    for(int i = 0; i < num_sub_hnsw; i++) {
        // 2 entries for each sub_hnsw
        offset_sub_hnsw.push_back(0); // start
        offset_sub_hnsw.push_back(0); // end with gap

        // 15 entries for each sub_hnsw
        offset_para.push_back(0); // idx->ntotal (idx_t) 0
        offset_para.push_back(0); // hnsw->levels (start,change size from here) 1  && get end calculate from here
        offset_para.push_back(0); // hnsw->offsets (start, change size from here) 2  && get end calculate from here
        offset_para.push_back(0); // hnsw->neighbors (start) 3 && get end calculate from here 
        offset_para.push_back(0); // hnsw->neighbors (end) 4 need delete
        offset_para.push_back(0); // hnsw->entry_point (storage_idx_t) 5
        offset_para.push_back(0); // hnsw->max_level (int) 6
        offset_para.push_back(0); // storage->header->ntotal (idx_t) 7
        offset_para.push_back(0); // storage->idxf->xb (start, change size from here) 8  && get end calculate from here
        offset_para.push_back(0); // hnsw->levels max boundary 9 == offsets start 2 need delete
        offset_para.push_back(0); // hnsw->offsets max boundary 10 == neighbors start 3 need delete
        offset_para.push_back(0); // hnsw->neighbors max boundary 11 == entry_point start 5 need delete
        offset_para.push_back(0); // storage->idxf->xb  max boundary 12 update in local only
        offset_para.push_back(0); // 2 bottom layers neighbors start 13
        offset_para.push_back(0); // 2 bottom layers neighbors end 14 update in local only


        
        overflow.push_back(0); // overflow gap start 0
        overflow.push_back(0); // overflow gap end 1
        overflow.push_back(0); // overflow gap max boundary 2
    }
        
    faiss::VectorIOWriter writer;
    size_t current_pos = 0;

    // Write header information
    writer(reinterpret_cast<const char*>(&d), sizeof(d), 1);
    current_pos += sizeof(d);
    writer(reinterpret_cast<const char*>(&num_sub_hnsw), sizeof(num_sub_hnsw), 1);
    current_pos += sizeof(num_sub_hnsw);
    writer(reinterpret_cast<const char*>(&meta_M), sizeof(meta_M), 1);
    current_pos += sizeof(meta_M);
    writer(reinterpret_cast<const char*>(&sub_M), sizeof(sub_M), 1);
    current_pos += sizeof(sub_M);
    writer(reinterpret_cast<const char*>(&num_meta), sizeof(num_meta), 1);
    current_pos += sizeof(num_meta);

    // Serialize meta HNSW index to get its size for overflow gaps
    size_t meta_start = current_pos;
    faiss::write_index(meta_hnsw, &writer);
    size_t meta_size = writer.data.size() - meta_start;
    current_pos += meta_size;
    std::cout << "Meta HNSW size: " << meta_size << std::endl;
    // Write number of sub indices
    size_t num_sub_indices = sub_hnsw.size();
    writer(reinterpret_cast<const char*>(&num_sub_indices), sizeof(num_sub_indices), 1);
    current_pos += sizeof(num_sub_indices);

    // Serialize each sub HNSW with overflow gaps between pairs
    for (size_t i = 0; i < sub_hnsw.size(); ++i) {
        offset_sub_hnsw[i * 2] = current_pos;
        faiss::write_dhnsw_index_init(sub_hnsw[i], &writer, offset_para, i, current_pos);
        offset_sub_hnsw[i * 2 + 1] = current_pos;
        // Add overflow gap between every second sub-HNSW 0-1, 2-3, 4-5, 6-7
        if(i % 2 == 0 ){
            size_t gap_size = 2 * meta_size; // Gap size is 2x meta HNSW size
            size_t gap_start = current_pos;
            size_t gap_end = current_pos + gap_size;
            // overflow for i
            overflow[i * 3] = gap_start;
            overflow[i * 3 + 1] = gap_start; // no overflow now
            overflow[i * 3 + 2] = gap_end; // max boundary 2
            
            // overflow for i+1
            if(i != sub_hnsw.size() - 1){
                overflow[(i+1) * 3] = gap_end;
                overflow[(i+1) * 3 + 1] = gap_end; // no overflow now
                overflow[(i+1) * 3 + 2] = gap_start; // max boundary 2 !!! important
            }
            std::vector<char> gap(gap_size, 0);
            writer(gap.data(), 1, gap_size);
            current_pos += gap_size;
        }
        
    }

    return writer.data;
    
}

DistributedHnsw DistributedHnsw::initial_deserialize_whole(
    const std::vector<uint8_t>& data,
    const std::vector<uint64_t>& offset_sub_hnsw,
    const std::vector<uint64_t>& offset_para,
    const std::vector<uint64_t>& overflow) {
    
    DistributedHnsw obj;
    faiss::VectorIOReader reader;
    reader.data.assign(data.begin(), data.end());
    size_t current_pos = 0;

    // Read header information
    std::cout << "Reading header..." << std::endl;
    reader(reinterpret_cast<char*>(&obj.d), sizeof(obj.d), 1); 
    current_pos += sizeof(obj.d);
    std::cout << "Read d: " << obj.d << std::endl;
    
    reader(reinterpret_cast<char*>(&obj.num_sub_hnsw), sizeof(obj.num_sub_hnsw), 1);
    current_pos += sizeof(obj.num_sub_hnsw);
    std::cout << "Read num_sub_hnsw: " << obj.num_sub_hnsw << std::endl;

    reader(reinterpret_cast<char*>(&obj.meta_M), sizeof(obj.meta_M), 1);
    current_pos += sizeof(obj.meta_M);
    std::cout << "Read meta_M: " << obj.meta_M << std::endl;
    
    reader(reinterpret_cast<char*>(&obj.sub_M), sizeof(obj.sub_M), 1);
    current_pos += sizeof(obj.sub_M);
    std::cout << "Read sub_M: " << obj.sub_M << std::endl;

    reader(reinterpret_cast<char*>(&obj.num_meta), sizeof(obj.num_meta), 1);
    current_pos += sizeof(obj.num_meta);
    std::cout << "Read num_meta: " << obj.num_meta << std::endl;

    // Read meta HNSW
    std::cout << "Reading meta_hnsw..." << std::endl;
    size_t meta_start = current_pos;
    obj.meta_hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(&reader));
    if (!obj.meta_hnsw) {
        throw std::runtime_error("Failed to read meta HNSW index");
    }

    size_t meta_size = reader.rp - meta_start;
    current_pos += meta_size;
    std::cout << "Meta HNSW read successfully, size: " << meta_size << std::endl;


    // Read number of sub indices
    size_t num_sub_indices;
    reader(&num_sub_indices, sizeof(num_sub_indices), 1);
    current_pos += sizeof(num_sub_indices);
    std::cout << "Number of sub indices: " << num_sub_indices << std::endl;
    obj.sub_hnsw.resize(num_sub_indices); 

    std::cout << "Processing sub_hnsw indices..." << std::endl;
    for (size_t i = 0; i < num_sub_indices; ++i) {

        std::cout << "Processing sub_hnsw " << i << std::endl;
        current_pos = offset_sub_hnsw[i * 2];   

        if (current_pos >= data.size()) {
            throw std::runtime_error("Buffer overflow while reading sub_hnsw");
        }
        std::cout << "offset_para values for index " << i << ":" << std::endl;
        for (int j = 0; j < 15; j++) {
            std::cout << "offset_para[" << (i * 15 + j) << "]: " << offset_para[i * 15 + j] << std::endl;
        }

        // Verify reader state
        std::cout << "Reader position: " << reader.rp << std::endl;
        std::cout << "Reader data size: " << reader.data.size() << std::endl;

        
        try {
            // Create a new IndexHNSWFlat object and assign it directly to obj.sub_hnsw[i]
            obj.sub_hnsw[i] = new faiss::IndexHNSWFlat();
            if (!obj.sub_hnsw[i]) {
                throw std::runtime_error("Failed to allocate new IndexHNSWFlat for sub_hnsw " + std::to_string(i));
            }
            
            std::cout << "Created new IndexHNSWFlat at " << obj.sub_hnsw[i] << std::endl;
            
            std::cout << "Calling read_dhnsw_index_init..." << std::endl;
            faiss::read_dhnsw_index_init(
                obj.sub_hnsw[i], 
                &reader,
                const_cast<std::vector<uint64_t>&>(offset_para),
                i,
                current_pos
            );
            std::cout << "Successfully read sub_hnsw " << i << std::endl;
            
            // Verify the object was properly initialized
            if (obj.sub_hnsw[i]->hnsw.entry_point < 0) {
                std::cout << "Warning: sub_hnsw[" << i << "] may not be properly initialized (entry_point: " 
                          << obj.sub_hnsw[i]->hnsw.entry_point << ")" << std::endl;
            } else {
                std::cout << "Verification: sub_hnsw[" << i << "]->hnsw.max_level = " 
                          << obj.sub_hnsw[i]->hnsw.max_level << std::endl;
                std::cout << "Verification: sub_hnsw[" << i << "]->hnsw.efSearch = " 
                          << obj.sub_hnsw[i]->hnsw.efSearch << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error reading sub_hnsw " << i << ": " << e.what() << std::endl;
            // Delete the partially initialized object to avoid memory leaks
            if (obj.sub_hnsw[i]) {
                delete obj.sub_hnsw[i];
                obj.sub_hnsw[i] = nullptr;
            }
            // Continue with next index instead of throwing
            continue;
        }
        

        //dealing with overflow
        // TODO: rewrite overflow relevant functions
    }
    std::cout << "Sub_hnsw indices read successfully" << std::endl;
    return obj;
}

void DistributedHnsw::append_levels_data(faiss::IndexHNSWFlat* index, const std::vector<uint8_t>& overflow_data) {
    if (!index || overflow_data.empty()) return;
    
    // Read size of levels vector from overflow data
    size_t num_levels;
    std::memcpy(&num_levels, overflow_data.data(), sizeof(size_t));
    
    // Calculate start position of actual data
    const uint8_t* data_start = overflow_data.data() + sizeof(size_t);
    
    // Calculate number of elements based on remaining data size
    size_t num_elements = (overflow_data.size() - sizeof(size_t)) / sizeof(int);
    
    // Append new levels
    std::vector<int> new_levels(reinterpret_cast<const int*>(data_start), 
                               reinterpret_cast<const int*>(data_start + num_elements * sizeof(int)));
    
    index->hnsw.levels.insert(index->hnsw.levels.end(), new_levels.begin(), new_levels.end());
}

void DistributedHnsw::append_offsets_data(faiss::IndexHNSWFlat* index, const std::vector<uint8_t>& overflow_data) {
    if (!index || overflow_data.empty()) return;
    
    // Calculate start position of actual data
    const uint8_t* data_start = overflow_data.data();
    
    // Calculate number of elements based on remaining data size
    size_t num_elements = (overflow_data.size() - sizeof(size_t)) / sizeof(size_t);
    
    // Append new offsets
    std::vector<size_t> new_offsets(reinterpret_cast<const size_t*>(data_start),
                                   reinterpret_cast<const size_t*>(data_start + num_elements * sizeof(size_t)));
    
    index->hnsw.offsets.insert(index->hnsw.offsets.end(), new_offsets.begin(), new_offsets.end());
}

void DistributedHnsw::append_neighbors_data(faiss::IndexHNSWFlat* index, const std::vector<uint8_t>& overflow_data) {
    if (!index || overflow_data.empty()) return;
    
    // Calculate start position of actual data
    const uint8_t* data_start = overflow_data.data();
    
    // Calculate number of elements based on remaining data size
    size_t num_elements = (overflow_data.size() - sizeof(size_t)) / sizeof(faiss::HNSW::storage_idx_t);
    
    // Append new neighbors
    std::vector<faiss::HNSW::storage_idx_t> new_neighbors(
        reinterpret_cast<const faiss::HNSW::storage_idx_t*>(data_start),
        reinterpret_cast<const faiss::HNSW::storage_idx_t*>(data_start + num_elements * sizeof(faiss::HNSW::storage_idx_t)));
    
    index->hnsw.neighbors.insert(index->hnsw.neighbors.end(), new_neighbors.begin(), new_neighbors.end());
}

void DistributedHnsw::append_xb_data(faiss::IndexHNSWFlat* index, const std::vector<uint8_t>& overflow_data, size_t d) {
    if (!index || overflow_data.empty()) return;

    const uint8_t* data_start = overflow_data.data();
    
    // Calculate number of elements based on remaining data size
    size_t num_elements = overflow_data.size() / sizeof(float);
    
    // Calculate number of vectors
    size_t num_vector = num_elements / d;
    
    // Get pointer to underlying storage
    faiss::IndexFlat* storage = dynamic_cast<faiss::IndexFlat*>(index->storage);
    if (!storage) return;
    
    // Append new vectors
    std::vector<float> new_vectors(reinterpret_cast<const float*>(data_start),
                                 reinterpret_cast<const float*>(data_start + num_elements * sizeof(float)));
    
    storage->add(num_vector, reinterpret_cast<const float*>(new_vectors.data()));
}

// Method to serialize the meta_hnsw
std::vector<uint8_t> DistributedHnsw::serialize_meta_hnsw() const {
    std::vector<uint8_t> buffer;
    std::ostringstream oss(std::ios::binary);

    faiss::VectorIOWriter writer;
    faiss::write_index(meta_hnsw, &writer);
    // Write the size of the serialized index
    size_t meta_index_size = writer.data.size();
    oss.write(reinterpret_cast<const char*>(&meta_index_size), sizeof(meta_index_size));
    // Write the serialized index data
    oss.write(reinterpret_cast<const char*>(writer.data.data()), meta_index_size);

    // Convert the ostringstream to a vector<uint8_t>
    std::string str = oss.str();
    buffer.assign(str.begin(), str.end());

    return buffer;
}

// Method to deserialize the meta_hnsw
faiss::IndexHNSWFlat* DistributedHnsw::deserialize_meta_hnsw(const std::vector<uint8_t>& data){
    faiss::IndexHNSWFlat* index = nullptr;
    std::istringstream iss(std::string(data.begin(), data.end()), std::ios::binary);
    {
    size_t meta_index_size;
    iss.read(reinterpret_cast<char*>(&meta_index_size), sizeof(meta_index_size));

    std::vector<uint8_t> index_data(meta_index_size);
    iss.read(reinterpret_cast<char*>(index_data.data()), meta_index_size);

    faiss::VectorIOReader reader;
    reader.data.assign(index_data.begin(), index_data.end());

    index = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(&reader));

    assert(index != nullptr && "Failed to deserialize meta_hnsw index"); 
    }

    return index;

}

// Method to serialize the sub-hnsw
std::vector<uint8_t> DistributedHnsw::serialize_sub_hnsw(int sub_index) const {
    std::vector<uint8_t> buffer;
    std::ostringstream oss(std::ios::binary);

    // 1. Serialize the sub_hnsw index
    {
        faiss::VectorIOWriter writer;
        faiss::write_index(sub_hnsw[sub_index], &writer);
        // Write the size of the serialized index
        size_t index_size = writer.data.size();
        oss.write(reinterpret_cast<const char*>(&index_size), sizeof(index_size));
        // Write the serialized index data
        oss.write(reinterpret_cast<const char*>(writer.data.data()), index_size);
    }

    // Convert the ostringstream to a vector<uint8_t>
    std::string str = oss.str();
    buffer.assign(str.begin(), str.end());

    return buffer;

}


// Method to deserialize the sub_hnsw
faiss::IndexHNSWFlat* DistributedHnsw::deserialize_sub_hnsw(const std::vector<uint8_t>& data) {
    faiss::IndexHNSWFlat* index = nullptr;
    // Deserialize the sub_hnsw index
    {
        faiss::VectorIOReader reader;
        reader.data.assign(data.begin(), data.end());
        index = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(&reader));

        assert(index != nullptr && "Failed to deserialize sub_hnsw index");
    }

    return index;
}

faiss::IndexHNSWFlat* LocalHnsw::deserialize_sub_hnsw_pipelined(const std::vector<uint8_t>& data) {
    faiss::IndexHNSWFlat* index = nullptr;
    if (data.empty()) {
        std::cerr << "[DESERIALIZATION ERROR] Input data is empty" << std::endl;
        return nullptr;
    }

    DirectMemoryIOReader reader(data); 

    try {
        index = faiss::read_index_HNSWFlat_optimized(&reader);
    } catch (const std::exception& e) {
        std::cerr << "[DESERIALIZATION ERROR] " << e.what() << std::endl;
        return nullptr;
    }
    
    if (index == nullptr) {
        std::cerr << "[DESERIALIZATION ERROR] read_index returned nullptr" << std::endl;
        return nullptr;
    }
    
    // Validate deserialized index to detect corruption from stale offsets
    if (index->ntotal < 0 || index->ntotal > 100000000) {
        std::cerr << "[VALIDATION ERROR] Invalid ntotal=" << index->ntotal 
                  << " (likely stale offsets)" << std::endl;
        delete index;
        return nullptr;
    }
    
    if (index->hnsw.levels.size() != static_cast<size_t>(index->ntotal)) {
        std::cerr << "[VALIDATION ERROR] HNSW levels.size()=" << index->hnsw.levels.size() 
                  << " != ntotal=" << index->ntotal << " (corrupted index)" << std::endl;
        delete index;
        return nullptr;
    }

    return index;
}
faiss::IndexHNSWFlat* LocalHnsw::deserialize_sub_hnsw_pipelined_(const std::vector<uint8_t>& data, int sub_idx) {
    faiss::IndexHNSWFlat* index = nullptr;
    if (data.empty()) {
        assert(false && "Input data for deserialization is empty");
        return nullptr;
    }

    DirectMemoryIOReader reader(data); 

    // Read offset arrays under shared lock to prevent race with init()
    std::vector<uint64_t> offset_para_tmp;
    std::vector<uint64_t> overflow_tmp;
    {
        std::shared_lock<std::shared_mutex> lock(epoch_mutex_);
        
        // Validate sub_idx bounds for offset_para_ (9 elements per sub-hnsw)
        size_t offset_para_base = sub_idx * 9;
        if (offset_para_base + 8 >= offset_para_.size()) {
            std::cerr << "Error: offset_para_ out of bounds for sub_idx " << sub_idx 
                      << " (need idx " << (offset_para_base + 8) << ", size=" << offset_para_.size() << ")" << std::endl;
            return nullptr;
        }
        
        // Validate sub_idx bounds for offset_subhnsw_ (2 elements per sub-hnsw)
        if (static_cast<size_t>(sub_idx * 2 + 1) >= offset_subhnsw_.size()) {
            std::cerr << "Error: offset_subhnsw_ out of bounds for sub_idx " << sub_idx << std::endl;
            return nullptr;
        }

        for (int i = 0; i < 9; i++) {
            offset_para_tmp.push_back(offset_para_[sub_idx * 9 + i] - offset_subhnsw_[sub_idx * 2]);
        }
        
        // overflow_ has 3 elements per sub-hnsw (not 9!)
        size_t overflow_base = sub_idx * 3;
        if (overflow_base + 2 < overflow_.size()) {
            for (int i = 0; i < 3; i++) {
                overflow_tmp.push_back(overflow_[overflow_base + i] - overflow_[overflow_base]);
            }
        } else {
            std::cerr << "Warning: overflow_ out of bounds for sub_idx " << sub_idx 
                      << ", using dummy values" << std::endl;
            overflow_tmp = {0, 0, 0};
        }
    }  // Release shared lock before deserialization (expensive operation)
    
    try {
        index = faiss::read_index_HNSWFlat_optimized_(&reader, offset_para_tmp, overflow_tmp);
    } catch (const std::exception& e) {
        std::cerr << "[DESERIALIZATION ERROR] sub_idx=" << sub_idx << ": " << e.what() << std::endl;
        return nullptr;
    }
    
    if (index == nullptr) {
        std::cerr << "Failed to deserialize sub_hnsw for sub_idx: " << sub_idx << std::endl;
        return nullptr;
    }
    
    // Validate deserialized index to detect corruption from stale offsets
    // This prevents heap-buffer-overflow in VisitedTable during search
    if (index->ntotal < 0 || index->ntotal > 100000000) {  // Sanity check: max 100M vectors per sub-index
        std::cerr << "[VALIDATION ERROR] sub_idx=" << sub_idx 
                  << " has invalid ntotal=" << index->ntotal 
                  << " (likely stale offsets, epoch mismatch)" << std::endl;
        delete index;
        return nullptr;
    }
    
    // Check HNSW structure consistency
    if (index->hnsw.levels.size() != static_cast<size_t>(index->ntotal)) {
        std::cerr << "[VALIDATION ERROR] sub_idx=" << sub_idx 
                  << " HNSW levels.size()=" << index->hnsw.levels.size() 
                  << " != ntotal=" << index->ntotal 
                  << " (corrupted index from stale offsets)" << std::endl;
        delete index;
        return nullptr;
    }

    return index;
}

// Method to deserialize the sub_hnsw with serialized_dhnsw_data and offset and sub_hnsw_tosearch
std::vector<faiss::IndexHNSWFlat*> DistributedHnsw::deserialize_sub_hnsw_batch(const std::vector<uint8_t>& data, const std::vector<uint64_t>& offset, std::vector<int> sub_hnsw_tosearch) {
    std::vector<faiss::IndexHNSWFlat*> sub_hnsw(sub_hnsw_tosearch.size(), nullptr);
    for(int i : sub_hnsw_tosearch){
        const std::vector<uint8_t> sub_data = std::vector<uint8_t>(data.begin() + offset[i * 4], data.begin() + offset[i * 4 + 1]);
        faiss::IndexHNSWFlat* sub_hnsw_tmp = deserialize_sub_hnsw(sub_data);
        sub_hnsw[i] = sub_hnsw_tmp;
    }
    return sub_hnsw;
}


// Method to serialize vector
std::vector<uint8_t> DistributedHnsw::serialize_offset(const std::vector<uint64_t>& vec) const {
    std::vector<uint8_t> buffer;
    std::ostringstream oss(std::ios::binary);

    // Write the size of the vector
    size_t vec_size = vec.size();
    oss.write(reinterpret_cast<const char*>(&vec_size), sizeof(vec_size));
    // Write the vector data
    oss.write(reinterpret_cast<const char*>(vec.data()), vec_size * sizeof(size_t));

    // Convert the ostringstream to a vector<uint8_t>
    std::string str = oss.str();
    buffer.assign(str.begin(), str.end());

    return buffer;
}

// Method to deserialize vector
std::vector<uint64_t> DistributedHnsw::deserialize_offset(const std::vector<uint8_t>& data){
    std::vector<uint64_t> vec;
    std::istringstream iss(std::string(data.begin(), data.end()), std::ios::binary);

    size_t vec_size;
    iss.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));

    vec.resize(vec_size);
    iss.read(reinterpret_cast<char*>(vec.data()), vec_size * sizeof(size_t));

    return vec;
}


std::vector<uint8_t> DistributedHnsw::serialize_initialsend(std::vector<uint64_t>& offset) const{
    std::vector<uint8_t> serialized_meta = serialize_meta_hnsw();
    std::vector<uint8_t> serialized_offset = serialize_offset(offset);
    std::vector<uint8_t> buffer = serialized_offset;
    buffer.insert(buffer.end(), serialized_meta.begin(), serialized_meta.end());
    return buffer;
}
    
std::vector<uint8_t> DistributedHnsw::serialize4client() const {
    std::vector<uint8_t> buffer;
    std::ostringstream oss(std::ios::binary);

    // 1. Serialize primitive data types
    oss.write(reinterpret_cast<const char*>(&d), sizeof(d));
    oss.write(reinterpret_cast<const char*>(&num_sub_hnsw), sizeof(num_sub_hnsw));
    oss.write(reinterpret_cast<const char*>(&meta_M), sizeof(meta_M));
    oss.write(reinterpret_cast<const char*>(&sub_M), sizeof(sub_M));
    oss.write(reinterpret_cast<const char*>(&num_meta), sizeof(num_meta));

    // 2. Serialize the meta_hnsw index
    {
        faiss::VectorIOWriter writer;
        faiss::write_index(meta_hnsw, &writer);
        // Write the size of the serialized index
        size_t meta_index_size = writer.data.size();
        oss.write(reinterpret_cast<const char*>(&meta_index_size), sizeof(meta_index_size));
        // Write the serialized index data
        oss.write(reinterpret_cast<const char*>(writer.data.data()), meta_index_size);
    }
    std::string str = oss.str();
    buffer.assign(str.begin(), str.end());

    return buffer;
}

void  DistributedHnsw::deserialize4client(
            const std::vector<uint8_t>& data,
            int& dim, int& num_sub_hnsw, int& meta_M, int& sub_M, int& num_meta,
            faiss::IndexHNSWFlat* meta_hnsw 
        ){

        std::istringstream iss(std::string(data.begin(), data.end()), std::ios::binary);

        // 1. Deserialize primitive data types
        iss.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        iss.read(reinterpret_cast<char*>(&num_sub_hnsw), sizeof(num_sub_hnsw));
        iss.read(reinterpret_cast<char*>(&meta_M), sizeof(meta_M));
        iss.read(reinterpret_cast<char*>(&sub_M), sizeof(sub_M));
        iss.read(reinterpret_cast<char*>(&num_meta), sizeof(num_meta));

        
        // 2. Deserialize the meta_hnsw index
        {
            size_t meta_index_size;
            iss.read(reinterpret_cast<char*>(&meta_index_size), sizeof(meta_index_size));

            std::vector<uint8_t> index_data(meta_index_size);
            iss.read(reinterpret_cast<char*>(index_data.data()), meta_index_size);

            faiss::VectorIOReader reader;

            reader.data.assign(index_data.begin(), index_data.end());

            meta_hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(faiss::read_index(&reader));

            assert(meta_hnsw != nullptr && "Failed to deserialize meta_hnsw index");
        } 
        
}

faiss::IndexHNSWFlat* DistributedHnsw::get_meta_hnsw() {
    return this->meta_hnsw;
}

std::vector<std::vector<dhnsw_idx_t>> DistributedHnsw::get_mapping() {
    return this->mapping;
}

std::vector<faiss::IndexHNSWFlat*> DistributedHnsw::get_sub_hnsw(std::vector<int> local_sub_hnsw_tag){
    std::vector<faiss::IndexHNSWFlat*> local_sub_hnsw;
    for (int i : local_sub_hnsw_tag) {
        local_sub_hnsw.push_back(this->sub_hnsw[i]);
    }
    return local_sub_hnsw; 
}

/*Local method*/

LocalHnsw::LocalHnsw()
    : d(0),
      num_sub_hnsw(0),
      meta_M(0),
      sub_M(0),
      meta_hnsw(nullptr),
      cache_(0)
    {
    // Do not allocate or initialize any resources
}
void LocalHnsw::init(){
    // Use AcquireEpochRead to get ALL metadata atomically for the current epoch
    // This avoids race conditions where separate RPCs could return data from different epochs
    auto epoch_read_info = dhnsw_client_->AcquireEpochRead("init_client");
    
    if (!epoch_read_info.success) {
        std::cerr << "Failed to acquire epoch read" << std::endl;
        return;
    }
    
    if (!epoch_read_info.has_metadata) {
        std::cerr << "Epoch " << epoch_read_info.epoch << " has no metadata" << std::endl;
        return;
    }
    
    if (epoch_read_info.serialized_meta_hnsw.empty()) {
        std::cerr << "Failed to retrieve meta_hnsw for epoch " << epoch_read_info.epoch << std::endl;
        return;
    }

    if (epoch_read_info.offset_subhnsw.empty()) {
        std::cerr << "Failed to retrieve offset for epoch " << epoch_read_info.epoch << std::endl;
        return;
    }

    if (epoch_read_info.mapping.empty()) {
        std::cerr << "Failed to retrieve mapping for epoch " << epoch_read_info.epoch << std::endl;
        return;
    }
    
    // Deserialize meta_hnsw before acquiring locks (expensive operation)
    faiss::IndexHNSWFlat* new_meta_hnsw = deserialize_meta_hnsw(epoch_read_info.serialized_meta_hnsw);

    // CRITICAL: Wait for in-flight batch operations to complete BEFORE updating state
    // This prevents workers from using stale offsets or accessing cleared cache entries
    // while they're still processing a batch with old data.
    std::unique_lock<std::mutex> batch_lock;
    if (pipelined_search_manager_ptr_) {
        std::cout << "[init] Waiting for in-flight batch to complete..." << std::endl;
        batch_lock = pipelined_search_manager_ptr_->get_processing_lock();
        std::cout << "[init] In-flight batch completed, proceeding with state update" << std::endl;
    }

    // Phase 2: Update all shared state atomically under exclusive lock
    // Note: batch_lock is still held here, ensuring no new batch starts until we're done
    {
        std::unique_lock<std::shared_mutex> lock(epoch_mutex_);
        
        // Update epoch and RDMA offset
        current_epoch_.store(epoch_read_info.epoch);
        current_rdma_base_offset_.store(epoch_read_info.rdma_base_offset);
        
        // Delete old meta_hnsw if exists
        if (meta_hnsw) {
            delete meta_hnsw;
            meta_hnsw = nullptr;
        }
        meta_hnsw = new_meta_hnsw;

        // Update offsets (convert uint64_t to match internal types)
        offset_subhnsw_.assign(epoch_read_info.offset_subhnsw.begin(), 
                               epoch_read_info.offset_subhnsw.end());
        offset_para_.assign(epoch_read_info.offset_para.begin(), 
                           epoch_read_info.offset_para.end());
        overflow_.assign(epoch_read_info.overflow.begin(), 
                        epoch_read_info.overflow.end());
        mapping = std::move(epoch_read_info.mapping);
        
        // Clear the LRU cache on reinit to avoid stale data
        cache_.clear();
    }
    // batch_lock is released here after state update is complete
    
    std::cout << "=== gRPC Service Parameters Debug ===" << std::endl;
    std::cout << "epoch: " << epoch_read_info.epoch << std::endl;
    std::cout << "rdma_base_offset: " << epoch_read_info.rdma_base_offset << std::endl;
    std::cout << "serialized_meta_hnsw size: " << epoch_read_info.serialized_meta_hnsw.size() << " bytes" << std::endl;
    std::cout << "offset size: " << epoch_read_info.offset_subhnsw.size() << " elements" << std::endl;
    std::cout << "offset_para size: " << epoch_read_info.offset_para.size() << " elements" << std::endl;
    std::cout << "overflow size: " << epoch_read_info.overflow.size() << " elements" << std::endl;
    std::cout << "mapping size: " << mapping.size() << " elements" << std::endl;
}

void LocalHnsw::set_meta_hnsw(faiss::IndexHNSWFlat* meta_hnsw_ptr){
    this->meta_hnsw = meta_hnsw_ptr;
    faiss::IndexFlat* storage = dynamic_cast<faiss::IndexFlat*>(meta_hnsw->storage);
    if (!storage) {
        std::cerr << "Failed to cast storage to IndexFlat." << std::endl;
    }

    // Print out the vectors
    // // std::cout << "Original vectors stored in the index:" << std::endl;
    // std::vector<float> retrieved_vector(d);
    // for (int64_t i = 0; i < 50; ++i) {
    //     storage->reconstruct(i, retrieved_vector.data());
    //     // std::cout << "Vector " << i << ": ";
    //     for (int64_t j = 0; j < d; ++j) {
    //         // std::cout << retrieved_vector[j] << " ";
    //     }
    //     // std::cout << std::endl;
    // }

}

void LocalHnsw::set_meta_ef_search(int ef_search){
    this->meta_hnsw->hnsw.efSearch = ef_search;
}


void LocalHnsw::set_local_sub_hnsw_tag(std::vector<int> local_sub_hnsw_tag){
    this->local_sub_hnsw_tag = local_sub_hnsw_tag;
}

void LocalHnsw::meta_search(const int n,const float* query, int K_meta, float* distances, dhnsw_idx_t* labels, std::vector<int>& sub_hnsw_tosearch) {
    meta_hnsw->search(n, query, K_meta, distances, labels);
    for(int i = 0; i < K_meta * n ; i++) {
        int label = labels[i];
        // std::cout << "label: " << label << std::endl;
        // if(!sub_hnsw_tosearch.count(part[label])){ //count difficulty
        sub_hnsw_tosearch.push_back(label);
        // std::cout << "sub_hnsw_tosearch: " << part[label] << std::endl;
        // }
    }
}

// std::pair<double, double> LocalHnsw::sub_search_each(const int n, const float* query, int K_meta, int K_sub, 
//                                 float* distances, dhnsw_idx_t* labels, 
//                                 std::vector<int>& sub_hnsw_tosearch, 
//                                 dhnsw_idx_t* sub_hnsw_tags,
//                                 int ef,
//                                 fetch_type flag) {
//     std::vector<std::tuple<float, dhnsw_idx_t, dhnsw_idx_t>> result;
//     std::unordered_map<int, std::unordered_set<int>> searchset; // Map sub_hnsw index to set of query indices
//     std::fill(distances, distances + n * K_sub, std::numeric_limits<float>::max());
//     std::fill(labels, labels + n * K_sub, -1);
//     std::fill(sub_hnsw_tags, sub_hnsw_tags + n * K_sub, -1);
//     double total_compute_time = 0;
//     double total_network_latency = 0;
//     // Build searchset: map each sub_hnsw to the set of query indices that need to search it
//     for (int i = 0; i < n * K_meta; i++) {
//         int query_idx = i / K_meta;
//         int sub_idx = sub_hnsw_tosearch[i];
//         searchset[sub_idx].insert(query_idx);
//     }

//     // Separate sub_hnsw indices into cached and uncached
//     std::vector<int> cached_sub_indices;
//     std::vector<int> uncached_sub_indices;
//     std::unordered_map<int, std::unordered_set<int>> cached_searchset;
//     std::unordered_map<int, std::unordered_set<int>> uncached_searchset;

//     for (const auto& entry : searchset) {
//         int sub_idx = entry.first;
//         const std::unordered_set<int>& query_indices_set = entry.second;

//         // Check if the sub_hnsw is in the cache
//         if (cache_.get(sub_idx) != nullptr) {
//             // Sub_hnsw is in cache
//             cached_sub_indices.push_back(sub_idx);
//             cached_searchset[sub_idx] = query_indices_set;
//         } else {
//             // Sub_hnsw not in cache
//             uncached_sub_indices.push_back(sub_idx);
//             uncached_searchset[sub_idx] = query_indices_set;
//         }
//     }

//     // First process cached sub_hnsw indices
//     for (int sub_idx : cached_sub_indices) {
//         const std::unordered_set<int>& query_indices_set = cached_searchset[sub_idx];

//         // Sub_hnsw is in cache
//         faiss::IndexHNSWFlat* sub_index = cache_.get(sub_idx);


//         // Proceed with search on sub_index
//         size_t num_queries = query_indices_set.size();
//         float* tmp_sub_distances = new float[K_sub * num_queries];
//         dhnsw_idx_t* tmp_sub_labels = new dhnsw_idx_t[K_sub * num_queries];
//         float* tmp_query = new float[d * num_queries];

//         int j = 0;
//         std::vector<int> query_indices(num_queries);
//         for (int tmp_query_idx : query_indices_set) {
//             query_indices[j] = tmp_query_idx;
//             std::copy(query + tmp_query_idx * d, query + (tmp_query_idx + 1) * d, tmp_query + j * d);
//             j++;
//         }
//         sub_index->hnsw.efSearch = ef;
//         sub_index->search(num_queries, tmp_query, K_sub, tmp_sub_distances, tmp_sub_labels);

//         // Merge the results
//         for (size_t j = 0; j < num_queries; j++) {
//             int tmp_query_idx = query_indices[j];
//             result.clear();

//             // Existing results for this query
//             for (int k = 0; k < K_sub; k++) {
//                 if (labels[tmp_query_idx * K_sub + k] != -1) {
//                     result.emplace_back(
//                         distances[tmp_query_idx * K_sub + k],
//                         labels[tmp_query_idx * K_sub + k],
//                         sub_hnsw_tags[tmp_query_idx * K_sub + k]
//                     );
//                 }
//             }

//             // New results from this sub_hnsw
//             for (int k = 0; k < K_sub; k++) {
//                 result.emplace_back(
//                     tmp_sub_distances[j * K_sub + k],
//                     tmp_sub_labels[j * K_sub + k],
//                     sub_idx
//                 );
//             }

//             // Sort the combined results
//             std::sort(result.begin(), result.end());

//             // Keep the top K_sub results
//             for (int k = 0; k < K_sub && k < result.size(); k++) {
//                 distances[tmp_query_idx * K_sub + k] = std::get<0>(result[k]);
//                 labels[tmp_query_idx * K_sub + k] = std::get<1>(result[k]);
//                 sub_hnsw_tags[tmp_query_idx * K_sub + k] = std::get<2>(result[k]);
//             }
//         }

//         delete[] tmp_sub_distances;
//         delete[] tmp_sub_labels;
//         delete[] tmp_query;
//     }

//     // Then process uncached sub_hnsw indices in batches
//     size_t batch_size = num_sub_hnsw/10; // Maximum number of sub_hnsw to fetch together
//     size_t num_uncached = uncached_sub_indices.size();
//     for (size_t i = 0; i < num_uncached; i += batch_size) {
//         size_t current_batch_size = std::min(batch_size, num_uncached - i);
//         std::vector<int> batch_sub_indices(uncached_sub_indices.begin() + i, uncached_sub_indices.begin() + i + current_batch_size);
//         std::vector<faiss::IndexHNSWFlat*> sub_indices;
//         auto start_fetch = high_resolution_clock::now();
//         if(flag == RDMA_DOORBELL){
//             // Fetch the batch of sub_hnsw from the server using RDMA doorbell
//             sub_indices = fetch_sub_hnsw_batch_with_doorbell(batch_sub_indices);
//         }
//         else if(flag == RDMA){
//             // Fetch the batch of sub_hnsw from the server using RDMA
//             // std::cout << "Fetching sub_hnsw using RDMA" << std::endl;
//             sub_indices = fetch_sub_hnsw_batch(batch_sub_indices);
            
//         }
//         else{
//             std::cerr << "Invalid flag" << std::endl;
//             return std::make_pair(-1.0, -1.0);
//         }
//         auto stop_fetch = high_resolution_clock::now();
//         auto duration_fetch = duration_cast<std::chrono::microseconds>(stop_fetch - start_fetch);
//         total_network_latency += (double)duration_fetch.count();
//         // std::cout << "Fetched " << sub_indices.size() << " sub_hnsw" << std::endl;
//         // Process each fetched sub_hnsw
//         for (size_t idx = 0; idx < batch_sub_indices.size(); ++idx) {
//             int sub_idx = batch_sub_indices[idx];
//             faiss::IndexHNSWFlat* sub_index = sub_indices[idx];

//             cache_.put(sub_idx, sub_index);

//             // Proceed with search on sub_index
//             const std::unordered_set<int>& query_indices_set = uncached_searchset[sub_idx];

//             size_t num_queries = query_indices_set.size();
//             float* tmp_sub_distances = new float[K_sub * num_queries];
//             dhnsw_idx_t* tmp_sub_labels = new dhnsw_idx_t[K_sub * num_queries];
//             float* tmp_query = new float[d * num_queries];

//             int j = 0;
//             std::vector<int> query_indices(num_queries);
//             for (int tmp_query_idx : query_indices_set) {
//                 query_indices[j] = tmp_query_idx;
//                 std::copy(query + tmp_query_idx * d, query + (tmp_query_idx + 1) * d, tmp_query + j * d);
//                 j++;
//             }
//             sub_index->hnsw.efSearch = ef;
//             auto start = high_resolution_clock::now();
//             sub_index->search(num_queries, tmp_query, K_sub, tmp_sub_distances, tmp_sub_labels);
//             auto stop = high_resolution_clock::now();
//             auto duration = duration_cast<std::chrono::microseconds>(stop - start);
//             total_compute_time += (double)duration.count();
//             // Merge the results
//             for (size_t j = 0; j < num_queries; j++) {
//                 int tmp_query_idx = query_indices[j];
//                 result.clear();

//                 // Existing results for this query
//                 for (int k = 0; k < K_sub; k++) {
//                     if (labels[tmp_query_idx * K_sub + k] != -1) {
//                         result.emplace_back(
//                             distances[tmp_query_idx * K_sub + k],
//                             labels[tmp_query_idx * K_sub + k],
//                             sub_hnsw_tags[tmp_query_idx * K_sub + k]
//                         );
//                     }
//                 }

//                 // New results from this sub_hnsw
//                 for (int k = 0; k < K_sub; k++) {
//                     result.emplace_back(
//                         tmp_sub_distances[j * K_sub + k],
//                         tmp_sub_labels[j * K_sub + k],
//                         sub_idx
//                     );
//                 }

//                 // Sort the combined results
//                 std::sort(result.begin(), result.end());

//                 // Keep the top K_sub results
//                 for (int k = 0; k < K_sub && k < result.size(); k++) {
//                     distances[tmp_query_idx * K_sub + k] = std::get<0>(result[k]);
//                     labels[tmp_query_idx * K_sub + k] = std::get<1>(result[k]);
//                     sub_hnsw_tags[tmp_query_idx * K_sub + k] = std::get<2>(result[k]);
//                 }
//             }

//             delete[] tmp_sub_distances;
//             delete[] tmp_sub_labels;
//             delete[] tmp_query;
//         }
//     }
//     return std::make_pair(total_compute_time, total_network_latency);
// }

// void LocalHnsw::sub_search_each_debug(const int n, const float* query, int K_meta, int K_sub, 
//                                 float* distances, dhnsw_idx_t* labels, 
//                                 std::vector<int>& sub_hnsw_tosearch, 
//                                 dhnsw_idx_t* sub_hnsw_tags,
//                                 int ef) {
//     std::vector<std::tuple<float, dhnsw_idx_t, dhnsw_idx_t>> result;
//     std::unordered_map<int, std::unordered_set<int>> searchset; // Map sub_hnsw index to set of query indices
//     std::fill(distances, distances + n * K_sub, std::numeric_limits<float>::max());
//     std::fill(labels, labels + n * K_sub, -1);
//     std::fill(sub_hnsw_tags, sub_hnsw_tags + n * K_sub, -1);

//     // Build searchset: map each sub_hnsw to the set of query indices that need to search it
//     for (int i = 0; i < n * K_meta; i++) {
//         int query_idx = i / K_meta;
//         int sub_idx = sub_hnsw_tosearch[i];
//         searchset[sub_idx].insert(query_idx);
//     }

//     // Separate sub_hnsw indices into cached and uncached
//     std::vector<int> cached_sub_indices;
//     std::vector<faiss::IndexHNSWFlat*> cached_sub_hnsw;
//     std::vector<int> uncached_sub_indices;
//     std::vector<faiss::IndexHNSWFlat*> uncached_sub_hnsw;
//     std::unordered_map<int, std::unordered_set<int>> cached_searchset;
//     std::unordered_map<int, std::unordered_set<int>> uncached_searchset;

//     for (const auto& entry : searchset) {
//         int sub_idx = entry.first;
//         const std::unordered_set<int>& query_indices_set = entry.second;

//         // Check if the sub_hnsw is in the cache
//         if (cache_.get(sub_idx) != nullptr) {
//             // Sub_hnsw is in cache
//             cached_sub_indices.push_back(sub_idx);
//             cached_searchset[sub_idx] = query_indices_set;
//             cached_sub_hnsw.push_back(cache_.get(sub_idx));
//         } else {
//             // Sub_hnsw not in cache
//             uncached_sub_indices.push_back(sub_idx);
//             uncached_searchset[sub_idx] = query_indices_set;
//             uncached_sub_hnsw.push_back(cache_.get(sub_idx));
//         }
//     }

//     // First process cached sub_hnsw indices
//     for (int i = 0; i < cached_sub_hnsw.size(); i++) {
//         const std::unordered_set<int>& query_indices_set = cached_searchset[cached_sub_indices[i]];

//         // Sub_hnsw is in cache
//         faiss::IndexHNSWFlat* sub_index = cached_sub_hnsw[i];


//         // Proceed with search on sub_index
//         size_t num_queries = query_indices_set.size();
//         float* tmp_sub_distances = new float[K_sub * num_queries];
//         dhnsw_idx_t* tmp_sub_labels = new dhnsw_idx_t[K_sub * num_queries];
//         float* tmp_query = new float[d * num_queries];

//         int j = 0;
//         std::vector<int> query_indices(num_queries);
//         for (int tmp_query_idx : query_indices_set) {
//             query_indices[j] = tmp_query_idx;
//             std::copy(query + tmp_query_idx * d, query + (tmp_query_idx + 1) * d, tmp_query + j * d);
//             j++;
//         }
//         sub_index->hnsw.efSearch = ef;
//         sub_index->search(num_queries, tmp_query, K_sub, tmp_sub_distances, tmp_sub_labels);

//         // Merge the results
//         for (size_t j = 0; j < num_queries; j++) {
//             int tmp_query_idx = query_indices[j];
//             result.clear();

//             // Existing results for this query
//             for (int k = 0; k < K_sub; k++) {
//                 if (labels[tmp_query_idx * K_sub + k] != -1) {
//                     result.emplace_back(
//                         distances[tmp_query_idx * K_sub + k],
//                         labels[tmp_query_idx * K_sub + k],
//                         sub_hnsw_tags[tmp_query_idx * K_sub + k]
//                     );
//                 }
//             }

//             // New results from this sub_hnsw
//             for (int k = 0; k < K_sub; k++) {
//                 result.emplace_back(
//                     tmp_sub_distances[j * K_sub + k],
//                     tmp_sub_labels[j * K_sub + k],
//                     cached_sub_indices[i]
//                 );
//             }

//             // Sort the combined results
//             std::sort(result.begin(), result.end());

//             // Keep the top K_sub results
//             for (int k = 0; k < K_sub && k < result.size(); k++) {
//                 distances[tmp_query_idx * K_sub + k] = std::get<0>(result[k]);
//                 labels[tmp_query_idx * K_sub + k] = std::get<1>(result[k]);
//                 sub_hnsw_tags[tmp_query_idx * K_sub + k] = std::get<2>(result[k]);
//             }
//         }

//         delete[] tmp_sub_distances;
//         delete[] tmp_sub_labels;
//         delete[] tmp_query;
//     }

//     // Then process uncached sub_hnsw indices in batches
//     size_t batch_size = num_sub_hnsw/10; // Maximum number of sub_hnsw to fetch together
//     size_t num_uncached = uncached_sub_indices.size();
//     for (size_t i = 0; i < num_uncached; i += batch_size) {
//         size_t current_batch_size = std::min(batch_size, num_uncached - i);
//         std::vector<int> batch_sub_indices(uncached_sub_indices.begin() + i, uncached_sub_indices.begin() + i + current_batch_size);
//         std::vector<faiss::IndexHNSWFlat*> sub_indices;
        
//         auto start2 = high_resolution_clock::now();
//         sub_indices = fetch_sub_hnsw_batch(batch_sub_indices);
//         auto stop2 = high_resolution_clock::now();
//         auto duration2 = duration_cast<std::chrono::microseconds>(stop2 - start2);
//         std::cout << "Time taken by normal: "
//             << duration2.count() << " microseconds" << std::endl;

//         auto start1 = high_resolution_clock::now();
//         sub_indices = fetch_sub_hnsw_batch_with_doorbell(batch_sub_indices);
//         auto stop1 = high_resolution_clock::now();
//         auto duration1 = duration_cast<std::chrono::microseconds>(stop1 - start1);
//         std::cout << "Time taken by doorbell: "
//             << duration1.count() << " microseconds" << std::endl;
            
//         // std::cout << "Fetched " << sub_indices.size() << " sub_hnsw" << std::endl;
//         // Process each fetched sub_hnsw
//         for (size_t idx = 0; idx < batch_sub_indices.size(); ++idx) {
//             int sub_idx = batch_sub_indices[idx];
//             faiss::IndexHNSWFlat* sub_index = sub_indices[idx];

//             cache_.put(sub_idx, sub_index);

//             // Proceed with search on sub_index
//             const std::unordered_set<int>& query_indices_set = uncached_searchset[sub_idx];

//             size_t num_queries = query_indices_set.size();
//             float* tmp_sub_distances = new float[K_sub * num_queries];
//             dhnsw_idx_t* tmp_sub_labels = new dhnsw_idx_t[K_sub * num_queries];
//             float* tmp_query = new float[d * num_queries];

//             int j = 0;
//             std::vector<int> query_indices(num_queries);
//             for (int tmp_query_idx : query_indices_set) {
//                 query_indices[j] = tmp_query_idx;
//                 std::copy(query + tmp_query_idx * d, query + (tmp_query_idx + 1) * d, tmp_query + j * d);
//                 j++;
//             }
//             sub_index->hnsw.efSearch = ef;
//             sub_index->search(num_queries, tmp_query, K_sub, tmp_sub_distances, tmp_sub_labels);

//             // Merge the results
//             for (size_t j = 0; j < num_queries; j++) {
//                 int tmp_query_idx = query_indices[j];
//                 result.clear();

//                 // Existing results for this query
//                 for (int k = 0; k < K_sub; k++) {
//                     if (labels[tmp_query_idx * K_sub + k] != -1) {
//                         result.emplace_back(
//                             distances[tmp_query_idx * K_sub + k],
//                             labels[tmp_query_idx * K_sub + k],
//                             sub_hnsw_tags[tmp_query_idx * K_sub + k]
//                         );
//                     }
//                 }

//                 // New results from this sub_hnsw
//                 for (int k = 0; k < K_sub; k++) {
//                     result.emplace_back(
//                         tmp_sub_distances[j * K_sub + k],
//                         tmp_sub_labels[j * K_sub + k],
//                         sub_idx
//                     );
//                 }

//                 // Sort the combined results
//                 std::sort(result.begin(), result.end());

//                 // Keep the top K_sub results
//                 for (int k = 0; k < K_sub && k < result.size(); k++) {
//                     distances[tmp_query_idx * K_sub + k] = std::get<0>(result[k]);
//                     labels[tmp_query_idx * K_sub + k] = std::get<1>(result[k]);
//                     sub_hnsw_tags[tmp_query_idx * K_sub + k] = std::get<2>(result[k]);
//                 }
//             }

//             delete[] tmp_sub_distances;
//             delete[] tmp_sub_labels;
//             delete[] tmp_query;
//         }
//     }
// }

// std::pair<double,double>
// LocalHnsw::sub_search_each_parallel(
//     int n,
//     const float* query,
//     int K_meta,
//     int K_sub,
//     float* distances,
//     dhnsw_idx_t* labels,
//     std::vector<int>& sub_hnsw_tosearch,
//     dhnsw_idx_t* sub_hnsw_tags,
//     int ef,
//     fetch_type flag)
// {
//     std::fill(distances,       distances    + n * K_sub,
//               std::numeric_limits<float>::max());
//     std::fill(labels,          labels       + n * K_sub, -1);
//     std::fill(sub_hnsw_tags,   sub_hnsw_tags+ n * K_sub, -1);

//     double total_compute_time = 0;
//     double total_network_latency = 0;

//     ///
//     // 1) Build a map: sub_index → set of query‐indices
//     //
//     std::unordered_map<int,std::unordered_set<int>> searchset;
//     searchset.reserve(n * 2);
//     for(int i = 0; i < n * K_meta; i++){
//         int qi = i / K_meta;
//         int sub_i = sub_hnsw_tosearch[i];
//         searchset[sub_i].insert(qi);
//     }

//     //
//     // 2) Split into cached vs uncached
//     //
//     std::vector<int> cached_idxs, uncached_idxs;
//     std::unordered_map<int,std::unordered_set<int>> cached_qs, uncached_qs;
//     cached_idxs.reserve(searchset.size());
//     uncached_idxs.reserve(searchset.size());
//     for(auto &e : searchset){
//         if(cache_.get(e.first) != nullptr){
//             cached_idxs.push_back(e.first);
//             cached_qs.emplace(e.first, std::move(e.second));
//         } else {
//             uncached_idxs.push_back(e.first);
//             uncached_qs.emplace(e.first, std::move(e.second));
//         }
//     }

//     //
//     // 3) Helper to run one sub‑index search, then merge its results
//     //
//     auto process_one = [&](int sub_i,
//                            const std::unordered_set<int>& qset,
//                            faiss::IndexHNSWFlat* sub_index)
//     {
//         if(qset.empty()) return;

//         // 3a) pack queries into contiguous tmp_query
//         int m = (int)qset.size();
//         std::vector<int> qidx; qidx.reserve(m);
//         float* tmp_query   = new float[m * d];
//         float* tmp_dist    = new float[m * K_sub];
//         dhnsw_idx_t* tmp_lbl = new dhnsw_idx_t[m * K_sub];

//         int j = 0;
//         for(int qi : qset){
//             qidx.push_back(qi);
//             std::copy(
//               query + qi*d,
//               query + qi*d + d,
//               tmp_query + j*d
//             );
//             j++;
//         }

//         // 3b) set ef and run FAISS search
//         sub_index->hnsw.efSearch = ef;
//         auto t0 = std::chrono::high_resolution_clock::now();
//         sub_index->search(m, tmp_query, K_sub, tmp_dist, tmp_lbl);
//         auto t1 = std::chrono::high_resolution_clock::now();
//         total_compute_time += std::chrono::duration<double,std::micro>(t1-t0).count();

//         // 3c) parallel merge into global arrays
//         #pragma omp parallel for schedule(static)
//         for(int t = 0; t < m; ++t){
//             int qi = qidx[t];
//             float* out_d = distances    + qi * K_sub;
//             dhnsw_idx_t* out_l = labels + qi * K_sub;
//             dhnsw_idx_t* out_t = sub_hnsw_tags + qi * K_sub;
//             float* in_d  = tmp_dist     + t   * K_sub;
//             dhnsw_idx_t* in_l  = tmp_lbl      + t   * K_sub;

//             // gather existing + new
//             std::vector<std::tuple<float,dhnsw_idx_t,dhnsw_idx_t>> merged;
//             merged.reserve(K_sub*2);

//             // existing
//             for(int k=0;k<K_sub;++k){
//                 if(out_l[k] != -1){
//                     merged.emplace_back(out_d[k],out_l[k],out_t[k]);
//                 }
//             }
//             // new from this sub-index
//             for(int k=0;k<K_sub;++k){
//                 merged.emplace_back(in_d[k],in_l[k], sub_i);
//             }

//             // keep top-K_sub
//             std::nth_element(
//               merged.begin(),
//               merged.begin()+K_sub,
//               merged.end(),
//               [](auto &a, auto &b){ return std::get<0>(a) < std::get<0>(b); }
//             );
//             merged.resize(K_sub);
//             std::sort(
//               merged.begin(), merged.end(),
//               [](auto &a, auto &b){ return std::get<0>(a) < std::get<0>(b); }
//             );

//             // write back
//             for(int k=0;k<K_sub;++k){
//                 out_d[k] = std::get<0>(merged[k]);
//                 out_l[k] = std::get<1>(merged[k]);
//                 out_t[k] = std::get<2>(merged[k]);
//             }
//         }

//         delete[] tmp_query;
//         delete[] tmp_dist;
//         delete[] tmp_lbl;
//     };

//     //
//     // 4) Process cached sub‑indexes
//     //
//     for(int sub_i : cached_idxs){
//         // update LRU
//         cache_.put(sub_i, cache_.get(sub_i));

//         process_one(sub_i, cached_qs[sub_i], cache_.get(sub_i));
//     }

//     //
//     // 5) Fetch & process uncached in batches of size K
//     //
//     int batch_doorbell = cache_.capacity_; // remember to change this
//     for(size_t i=0; i<uncached_idxs.size(); i += batch_doorbell){
//         size_t batch_sz = std::min((size_t)batch_doorbell, uncached_idxs.size()-i);
//         std::vector<int> batch(uncached_idxs.begin()+i,
//                                uncached_idxs.begin()+i+batch_sz);

//         // fetch via RDMA or doorbell
//         auto t0 = std::chrono::high_resolution_clock::now();
//         auto fetched = (flag==RDMA_DOORBELL)
//           ? fetch_sub_hnsw_batch_with_doorbell(batch)
//           : fetch_sub_hnsw_batch(batch);
//         auto t1 = std::chrono::high_resolution_clock::now();
//         total_network_latency += std::chrono::duration<double,std::micro>(t1-t0).count();

//         // insert into cache & process
//         for(size_t j=0; j<batch_sz; ++j){
//             int sub_i = batch[j];
//             cache_.put(sub_i, fetched[j]);

//             process_one(sub_i, uncached_qs[sub_i], fetched[j]);
//         }
//     }

//     return { total_compute_time, total_network_latency };
// }


// std::pair<double, double> LocalHnsw::sub_search_each_naive(const int n, const float* query, int K_meta, int K_sub, 
//                                       float* distances, dhnsw_idx_t* labels, 
//                                       std::vector<int>& sub_hnsw_tosearch, 
//                                       dhnsw_idx_t* sub_hnsw_tags, 
//                                       int ef,
//                                       fetch_type flag) {
//     // Initialize distances and labels
//     std::fill(distances, distances + n * K_sub, std::numeric_limits<float>::max());
//     std::fill(labels, labels + n * K_sub, -1);
//     std::fill(sub_hnsw_tags, sub_hnsw_tags + n * K_sub, -1);
//     double total_compute_time = 0;
//     double total_network_latency = 0;
//     // For each query
//     for (int i = 0; i < n; i++) {

//         std::vector<std::tuple<float, dhnsw_idx_t, dhnsw_idx_t>> result;
//         // std::cout<< "Query " << i << ": " << std::endl;
//         // For each sub-index to search
//         for (int j = 0; j < K_meta; j++) {
//             int sub_idx = sub_hnsw_tosearch[i * K_meta + j];
//             faiss::IndexHNSWFlat* sub_index = nullptr;

//             // Check if the sub-index is in cache
//             sub_index = cache_.get(sub_idx);
//             if (sub_index != nullptr) {
//                 // Sub-index is in cache
               
//             } else {
//                 // Sub-index not in cache, fetch it
//                 // std::cout<< "Fetching sub-index " << sub_idx << std::endl;
//                 auto start_fetch = high_resolution_clock::now();
//                 if(flag == RDMA){
//                     sub_index = fetch_sub_hnsw(sub_idx);
//                 }
//                 else if(flag == RPC){
//                     sub_index = fetch_sub_hnsw_grpc(sub_idx);
//                 }
//                 else{
//                     std::cerr << "Invalid flag" << std::endl;
//                     return std::make_pair(-1.0, -1.0);
//                 }
//                 auto stop_fetch = high_resolution_clock::now();
//                 auto duration_fetch = duration_cast<std::chrono::microseconds>(stop_fetch - start_fetch);
//                 total_network_latency += (double)duration_fetch.count();
//                 if (sub_index == nullptr) {
//                     std::cerr << "Failed to fetch sub-index " << sub_idx << std::endl;
//                     continue;
//                 }
//                 // If cache is full, remove the least recently used sub-index
//                 cache_.put(sub_idx, sub_index);
//             }

//             // Perform search on the sub-index for the current query
//             float tmp_sub_distances[K_sub];
//             dhnsw_idx_t tmp_sub_labels[K_sub];
//             sub_index->hnsw.efSearch = ef;
            
//             auto start = high_resolution_clock::now();
//             sub_index->search(1, query + i * d, K_sub, tmp_sub_distances, tmp_sub_labels);
//             auto stop = high_resolution_clock::now();
//             auto duration = duration_cast<std::chrono::microseconds>(stop - start);
//             total_compute_time += (double)duration.count();
//             // Collect results
//             for (int k = 0; k < K_sub; k++) {
//                 result.emplace_back(tmp_sub_distances[k], tmp_sub_labels[k], sub_idx);
//             }
//         }

//         // Sort all results for the current query and keep the top K_sub
//         std::sort(result.begin(), result.end());
//         for (int k = 0; k < K_sub && k < result.size(); k++) {
//             distances[i * K_sub + k] = std::get<0>(result[k]);
//             labels[i * K_sub + k] = std::get<1>(result[k]);
//             sub_hnsw_tags[i * K_sub + k] = std::get<2>(result[k]);
//         }
//     }
//     return std::make_pair(total_compute_time, total_network_latency);
// }


void LocalHnsw::set_rdma_qp(std::shared_ptr<rdmaio::qp::RC> qp, const rdmaio::rmem::RegAttr& remote_attr, const rdmaio::Arc<rdmaio::rmem::RegHandler>& local_mr) {
    qp->bind_remote_mr(remote_attr);
    qp->bind_local_mr(local_mr->get_reg_attr().value());
    this->qp = qp.get();
    this->qp_shared = qp; 
    // std::cout << "set rdma_qp" << std::endl;
}

void LocalHnsw::set_remote_attr(const rmem::RegAttr& remote_attr) {
    this->remote_attr = remote_attr;
}

void LocalHnsw::set_local_mr(const rdmaio::Arc<rdmaio::rmem::RegHandler>& local_mr, const Arc<RMem>& local_mem) {
    this->local_mr = local_mr;
    this->local_mem_ = local_mem;
}

void LocalHnsw::set_offset_subhnsw(const std::vector<uint64_t>& offset_subhnsw) {
    // Note: caller must hold epoch_mutex_ if thread safety is needed
    this->offset_subhnsw_ = offset_subhnsw;
    // std::cout << "Offset " << std::endl;
    // Print the offset values
    // for (const auto& val : offset_subhnsw) {
    //     std::cout << val << " ";
    // }
}

void LocalHnsw::set_offset_para(const std::vector<uint64_t>& offset_para) {
    this->offset_para_ = offset_para;
    // std::cout << "Offset " << std::endl;
    // Print the offset values
    // for (const auto& val : offset_para) {
    //     std::cout << val << " ";
    // }
}

void LocalHnsw::set_offset_overflow(const std::vector<uint64_t>& offset_overflow) {
    this->overflow_ = offset_overflow;
    // std::cout << "Offset " << std::endl;
    // Print the offset values
    // for (const auto& val : offset_overflow) {
    //     std::cout << val << " ";
    // }
}
std::shared_ptr<faiss::IndexHNSWFlat> LocalHnsw::fetch_sub_hnsw(int sub_idx) {
    // Validate sub_idx
    if (sub_idx < 0 || sub_idx >= num_sub_hnsw) {
        std::cerr << "Invalid sub_idx: " << sub_idx << std::endl;
        return nullptr;
    }

    // Calculate positions and length under shared lock
    uint64_t base_offset = current_rdma_base_offset_.load(std::memory_order_acquire);
    uint64_t rel_start, rel_end;
    {
        std::shared_lock<std::shared_mutex> lock(epoch_mutex_);
        if (static_cast<size_t>(sub_idx * 2 + 1) >= offset_subhnsw_.size()) {
            std::cerr << "Error: offset_subhnsw_ out of bounds in fetch_sub_hnsw for sub_idx " << sub_idx << std::endl;
            return nullptr;
        }
        rel_start = offset_subhnsw_[sub_idx * 2];
        rel_end = offset_subhnsw_[sub_idx * 2 + 1];
    }
    uint64_t start_pos = base_offset + rel_start;
    u32 length = static_cast<u32>(rel_end - rel_start);

    // std::cout << "Start pos: " << start_pos << " (base=" << base_offset << ", rel=" << rel_start << ")" << std::endl;
    // std::cout << "Length: " << length << std::endl;

    // Access local memory
    if (!local_mem_) {
        std::cerr << "Error: local_mem_ is null" << std::endl;
        return nullptr;
    }

    if (!local_mem_->raw_ptr) {
        std::cerr << "Error: local_mem_->raw_ptr is null" << std::endl;
        return nullptr;
    }

    uint8_t* recv_buffer = reinterpret_cast<uint8_t*>(local_mem_->raw_ptr);
    size_t local_mem_size = local_mem_->sz;

    // std::cout << "Local mem size: " << local_mem_size << std::endl;

    // Ensure the local buffer is large enough
    if (length > local_mem_size) {
        std::cerr << "Data length exceeds local buffer size" << std::endl;
        return nullptr;
    }

    // Perform RDMA read to fetch the data
    auto res_s = qp->send_normal(
        {
            .op = IBV_WR_RDMA_READ,
            .flags = IBV_SEND_SIGNALED,
            .len = length,
            .wr_id = 0
        },
        {
            .local_addr = reinterpret_cast<RMem::raw_ptr_t>(recv_buffer),
            .remote_addr = start_pos,
            .imm_data = 0
        }
    );

    RDMA_ASSERT(res_s == IOCode::Ok) << "RDMA read failed: " << res_s.desc;
    // std::cout << "RDMA read initiated for sub_idx " << sub_idx << std::endl;

    // Wait for completion
    auto res_p = qp->wait_one_comp();
    RDMA_ASSERT(res_p == IOCode::Ok) << "RDMA read completion failed " << std::endl;
    // std::cout << "RDMA read completed for sub_idx " << sub_idx << std::endl;

    // Deserialize the sub-index
    std::vector<uint8_t> serialized_data(recv_buffer, recv_buffer + length);
    faiss::IndexHNSWFlat* sub_index = deserialize_sub_hnsw_pipelined_(serialized_data,sub_idx);

    if (!sub_index) {
        std::cerr << "Failed to deserialize sub_hnsw for sub_idx: " << sub_idx << std::endl;
        return nullptr;
    }

    // Optionally clear the buffer
    memset(recv_buffer, 0, length);

    return std::shared_ptr<faiss::IndexHNSWFlat>(sub_index);
}

// faiss::IndexHNSWFlat* LocalHnsw::fetch_sub_hnsw_grpc(int sub_idx){

//     std::vector<uint8_t> serialized_sub_hnsw = dhnsw_client_->GetSub_hnsw(sub_idx);
//     if (serialized_sub_hnsw.empty()) {
//             std::cerr << "Failed to fetch sub_hnsw with index " << sub_idx << std::endl;
//             return nullptr;
//         }
//     faiss::IndexHNSWFlat* sub_index = DistributedHnsw::deserialize_sub_hnsw(serialized_sub_hnsw);
//         if (!sub_index) {
//             std::cerr << "Failed to deserialize sub_hnsw with index " << sub_idx << std::endl;
//             return nullptr;
//         }

//     return sub_index;
// }

// void LocalHnsw::sub_search_each_test(const int n, const float* query, int K_meta, int K_sub, 
//                                 float* distances, dhnsw_idx_t* labels, 
//                                 std::vector<int>& sub_hnsw_tosearch, dhnsw_idx_t* sub_hnsw_tags) {
//     std::vector<std::tuple<float, dhnsw_idx_t, dhnsw_idx_t>> result;
//     std::unordered_map<int, std::unordered_set<int>> searchset; // Map sub_hnsw index to set of query indices
//     std::fill(distances, distances + n * K_sub, std::numeric_limits<float>::max());
//     std::fill(labels, labels + n * K_sub, -1);
//     std::fill(sub_hnsw_tags, sub_hnsw_tags + n * K_sub, -1);

//     // Build searchset: map each sub_hnsw to the set of query indices that need to search it
//     for (int i = 0; i < n * K_meta; i++) {
//         int query_idx = i / K_meta;
//         int sub_idx = sub_hnsw_tosearch[i];
//         searchset[sub_idx].insert(query_idx);
//     }

//     // Separate sub_hnsw indices into cached and uncached
//     std::vector<int> cached_sub_indices;
//     std::vector<faiss::IndexHNSWFlat*> cached_sub_hnsw;
//     std::vector<int> uncached_sub_indices;
//     std::vector<faiss::IndexHNSWFlat*> uncached_sub_hnsw;
//     std::unordered_map<int, std::unordered_set<int>> cached_searchset;
//     std::unordered_map<int, std::unordered_set<int>> uncached_searchset;

//     for (const auto& entry : searchset) {
//         int sub_idx = entry.first;
//         const std::unordered_set<int>& query_indices_set = entry.second;

//         // Check if the sub_hnsw is in the cache
//         if (cache_.get(sub_idx) != nullptr) {
//             // Sub_hnsw is in cache
//             cached_sub_indices.push_back(sub_idx);
//             cached_searchset[sub_idx] = query_indices_set;
//             cached_sub_hnsw.push_back(cache_.get(sub_idx));
//         } else {
//             // Sub_hnsw not in cache
//             uncached_sub_indices.push_back(sub_idx);
//             uncached_searchset[sub_idx] = query_indices_set;
//             uncached_sub_hnsw.push_back(cache_.get(sub_idx));
//         }
//     }

//     // First process cached sub_hnsw indices
//     for (int i = 0; i < cached_sub_hnsw.size(); i++) {
//         const std::unordered_set<int>& query_indices_set = cached_searchset[cached_sub_indices[i]];

//         // Sub_hnsw is in cache
//         faiss::IndexHNSWFlat* sub_index = cached_sub_hnsw[i];


//         // Proceed with search on sub_index
//         size_t num_queries = query_indices_set.size();
//         float* tmp_sub_distances = new float[K_sub * num_queries];
//         dhnsw_idx_t* tmp_sub_labels = new dhnsw_idx_t[K_sub * num_queries];
//         float* tmp_query = new float[d * num_queries];

//         int j = 0;
//         std::vector<int> query_indices(num_queries);
//         for (int tmp_query_idx : query_indices_set) {
//             query_indices[j] = tmp_query_idx;
//             std::copy(query + tmp_query_idx * d, query + (tmp_query_idx + 1) * d, tmp_query + j * d);
//             j++;
//         }

//         sub_index->search(num_queries, tmp_query, K_sub, tmp_sub_distances, tmp_sub_labels);

//         // Merge the results
//         for (size_t j = 0; j < num_queries; j++) {
//             int tmp_query_idx = query_indices[j];
//             result.clear();

//             // Existing results for this query
//             for (int k = 0; k < K_sub; k++) {
//                 if (labels[tmp_query_idx * K_sub + k] != -1) {
//                     result.emplace_back(
//                         distances[tmp_query_idx * K_sub + k],
//                         labels[tmp_query_idx * K_sub + k],
//                         sub_hnsw_tags[tmp_query_idx * K_sub + k]
//                     );
//                 }
//             }

//             // New results from this sub_hnsw
//             for (int k = 0; k < K_sub; k++) {
//                 result.emplace_back(
//                     tmp_sub_distances[j * K_sub + k],
//                     tmp_sub_labels[j * K_sub + k],
//                     cached_sub_indices[i]
//                 );
//             }

//             // Sort the combined results
//             std::sort(result.begin(), result.end());

//             // Keep the top K_sub results
//             for (int k = 0; k < K_sub && k < result.size(); k++) {
//                 distances[tmp_query_idx * K_sub + k] = std::get<0>(result[k]);
//                 labels[tmp_query_idx * K_sub + k] = std::get<1>(result[k]);
//                 sub_hnsw_tags[tmp_query_idx * K_sub + k] = std::get<2>(result[k]);
//             }
//         }

//         delete[] tmp_sub_distances;
//         delete[] tmp_sub_labels;
//         delete[] tmp_query;
//     }

//     // Then process uncached sub_hnsw indices
//     int batch_doorbell = 1; // remember to change this
//     for(size_t i=0; i<uncached_sub_indices.size(); i += batch_doorbell){
//         size_t batch_sz = std::min((size_t)batch_doorbell, uncached_sub_indices.size()-i);
//         std::vector<int> batch(uncached_sub_indices.begin()+i, uncached_sub_indices.begin()+i+batch_sz);
//         std::vector<faiss::IndexHNSWFlat*> fetched = fetch_sub_hnsw_batch(batch);

//         for (int i = 0; i < batch_sz; i++) {
//             faiss::IndexHNSWFlat* sub_index = fetched[i];
//             size_t num_queries = uncached_searchset[uncached_sub_indices[i]].size();
//             float* tmp_sub_distances = new float[K_sub * num_queries];
//             dhnsw_idx_t* tmp_sub_labels = new dhnsw_idx_t[K_sub * num_queries];
//             float* tmp_query = new float[d * num_queries];

//         int j = 0;
//         std::vector<int> query_indices(num_queries);
//         for (int tmp_query_idx : uncached_searchset[uncached_sub_indices[i]]) {
//             query_indices[j] = tmp_query_idx;
//             std::copy(query + tmp_query_idx * d, query + (tmp_query_idx + 1) * d, tmp_query + j * d);
//             j++;
//         }

//         sub_index->search(num_queries, tmp_query, K_sub, tmp_sub_distances, tmp_sub_labels);

//         for (size_t j = 0; j < num_queries; j++) {
//             int tmp_query_idx = query_indices[j];
//             result.clear();

//             for (int k = 0; k < K_sub; k++) {
//                 if (labels[tmp_query_idx * K_sub + k] != -1) {
//                     result.emplace_back(
//                         distances[tmp_query_idx * K_sub + k],
//                         labels[tmp_query_idx * K_sub + k],
//                         uncached_sub_indices[i]
//                     );
//                 }
//             }

//             std::sort(result.begin(), result.end());
//         }
//     }
// }
                                // }
void LocalHnsw::set_mapping(const std::vector<std::vector<dhnsw_idx_t>>& mapping) {
        this->mapping = mapping;
        // std::cout << "Mapping set with size: " << mapping.size() << std::endl;
}

int LocalHnsw::get_original_index(int sub_idx, int local_label) {
    // Acquire shared lock to prevent concurrent init() from modifying mapping
    std::shared_lock<std::shared_mutex> lock(epoch_mutex_);
    
    if (sub_idx < 0 || sub_idx >= static_cast<int>(mapping.size())) {
        return -1;
    }
    if (local_label < 0 || local_label >= static_cast<int>(mapping[sub_idx].size())) {
        return -1;
    }
    return mapping[sub_idx][local_label];
}

std::vector<faiss::IndexHNSWFlat*> LocalHnsw::fetch_sub_hnsw_batch(const std::vector<int>& sub_indices) {
    // std::cout << "Fetching sub hnsw batch without doorbell" << std::endl;
    size_t batch_size = sub_indices.size();
    std::vector<faiss::IndexHNSWFlat*> sub_hnsw_list(batch_size, nullptr);

    // Prepare vectors for RDMA operations
    std::vector<uint64_t> sizes(batch_size);
    std::vector<uint64_t> local_offsets(batch_size);
    std::vector<uint64_t> remote_offsets(batch_size);
    size_t total_size = 0;

    // Get base offset for this epoch
    uint64_t base_offset = current_rdma_base_offset_.load(std::memory_order_acquire);

    // Determine sizes and offsets under shared lock
    {
        std::shared_lock<std::shared_mutex> lock(epoch_mutex_);
        for (size_t i = 0; i < batch_size; ++i) {
            int sub_idx = sub_indices[i];
            if (static_cast<size_t>(sub_idx * 2 + 1) >= offset_subhnsw_.size()) {
                std::cerr << "Error: offset_subhnsw_ out of bounds in fetch_batch for sub_idx " << sub_idx << std::endl;
                return {};
            }
            size_t rel_start = offset_subhnsw_[sub_idx * 2 + 0];
            size_t rel_end = offset_subhnsw_[sub_idx * 2 + 1];
            size_t size = rel_end - rel_start;

            sizes[i] = size;
            remote_offsets[i] = base_offset + rel_start;
            local_offsets[i] = total_size;
            total_size += size;
        }
    }

    // Access local memory
    uint8_t* recv_buffer = reinterpret_cast<uint8_t*>(local_mem_->raw_ptr);
    size_t local_mem_size = local_mem_->sz; 
    if (!local_mem_) {
    std::cerr << "Error: local_mem_ is null." << std::endl;
    return {};
    }
    if (!local_mem_->raw_ptr) {
        std::cerr << "Error: local_mem_->raw_ptr is null." << std::endl;
        return {};
    }
    if (!qp) {
    std::cerr << "Error: qp (Queue Pair) is null." << std::endl;
    return {};
    }
    // Ensure the local memory region has enough space
    if (total_size > local_mem_size) {
        RDMA_LOG(ERROR) << "Local MR size is insufficient for batch fetch. Required size: " << total_size;
        return {};
    }

    // Perform RDMA reads for each sub-index
    for (size_t i = 0; i < batch_size; ++i) {
        // std::cout << "Fetching sub-index " << sub_indices[i] << std::endl;
        // std::cout << "Remote offset: " << remote_offsets[i] << ", Size: " << sizes[i] << ", Local offset: " << local_offsets[i] << std::endl;

        auto res_s = qp->send_normal(
            {
                .op = IBV_WR_RDMA_READ,
                .flags = IBV_SEND_SIGNALED,
                .len = static_cast<uint32_t>(sizes[i]),
                .wr_id = 0
            },
            {
                .local_addr = reinterpret_cast<RMem::raw_ptr_t>(recv_buffer + local_offsets[i]),
                .remote_addr = remote_offsets[i],
                .imm_data = 0
            }
        );

        RDMA_ASSERT(res_s == IOCode::Ok) << "Failed to post RDMA read for sub_index " << sub_indices[i] << ": " << res_s.desc;

        // Wait for completion
        auto res_p = qp->wait_one_comp();
        if (res_p != IOCode::Ok) {
            RDMA_LOG(ERROR) << "Failed to get RDMA read completion for sub_index " << std::endl;
            return {};
   
        }
    }
    for (size_t i = 0; i < batch_size; i++)
    {
        uint8_t* data_ptr = recv_buffer + local_offsets[i];
        size_t data_size = sizes[i];

        // Deserialize the sub-index
        // std::cout << "Deserializing sub-index " << sub_indices[i] << std::endl;
        std::vector<uint8_t> serialized_data(data_ptr, data_ptr + data_size);
        faiss::IndexHNSWFlat* sub_index = DistributedHnsw::deserialize_sub_hnsw(serialized_data);

        if (sub_index == nullptr) {
            RDMA_LOG(ERROR) << "Failed to deserialize sub_hnsw index for sub_idx: " << sub_indices[i];
            return {};
        }

        sub_hnsw_list[i] = sub_index;
    }
    return sub_hnsw_list;
    
}

std::vector<faiss::IndexHNSWFlat*> LocalHnsw::fetch_sub_hnsw_batch_with_doorbell(const std::vector<int>& sub_indices) {
    size_t batch_size = sub_indices.size();
    std::vector<faiss::IndexHNSWFlat*> sub_hnsw_list(batch_size, nullptr);

    // Prepare vectors for RDMA operations
    std::vector<uint64_t> sizes(batch_size);
    std::vector<uint64_t> local_offsets(batch_size);
    std::vector<uint64_t> remote_offsets(batch_size);
    size_t total_size = 0;

    // Get base offset for this epoch
    uint64_t base_offset = current_rdma_base_offset_.load(std::memory_order_acquire);

    // Determine sizes and offsets under shared lock
    {
        std::shared_lock<std::shared_mutex> lock(epoch_mutex_);
        for (size_t i = 0; i < batch_size; ++i) {
            int sub_idx = sub_indices[i];
            if (static_cast<size_t>(sub_idx * 2 + 1) >= offset_subhnsw_.size()) {
                std::cerr << "Error: offset_subhnsw_ out of bounds in fetch_batch_doorbell for sub_idx " << sub_idx << std::endl;
                return {};
            }
            size_t rel_start = offset_subhnsw_[sub_idx * 2 + 0];
            size_t rel_end   = offset_subhnsw_[sub_idx * 2 + 1];
            size_t size      = rel_end - rel_start;
            sizes[i] = size;
            remote_offsets[i] = base_offset + rel_start;
            local_offsets[i]  = total_size;
            total_size += size;
        }
    }

    if (!local_mem_ || !local_mem_->raw_ptr) {
        std::cerr << "Error: local memory not properly allocated." << std::endl;
        return {};
    }
    uint8_t* recv_buffer = reinterpret_cast<uint8_t*>(local_mem_->raw_ptr);
    size_t local_mem_size = local_mem_->sz;
    if (total_size > local_mem_size) {
        RDMA_LOG(ERROR) << "Local MR size insufficient for batch fetch. Required size: " << total_size;
        return {};
    }

    // Check against the maximum allowed batch size.
    const int max_batch_size = num_sub_hnsw/10;  
    if (batch_size > max_batch_size) {
        std::cerr << "Batch size exceeds the maximum allowed batch size." << std::endl;
        return {};
    }

    // Prepare the batched RDMA read operations.
    BenchOp<> ops[max_batch_size];
    u32 lkey = local_mr->get_reg_attr().value().key;
    u32 rkey = remote_attr.key;

    for (int i = 0; i < batch_size; ++i) {
        ops[i].set_type(0);  // RDMA_READ
        // Compute the absolute remote address by adding the base remote address.
        uint64_t remote_addr = reinterpret_cast<uint64_t>(remote_attr.buf) + remote_offsets[i];
        // Note: init_rbuf expects a pointer to the remote memory address.
        ops[i].init_rbuf(reinterpret_cast<u64*>(remote_addr), rkey);
        ops[i].init_lbuf(reinterpret_cast<u64*>(recv_buffer + local_offsets[i]), sizes[i], lkey);
        ops[i].set_wrid(i);
        ops[i].set_flags(0);  // Clear any flags.
        if (i != 0) {
            ops[i - 1].set_next(&ops[i]);  // Chain the operations.
        }
    }
    // Only the last op is signaled to reduce overhead.
    ops[batch_size - 1].set_flags(IBV_SEND_SIGNALED);

    // Execute the entire batch with one doorbell ring.
    auto res_s = ops[0].execute_batch(qp_shared);
    if (res_s != IOCode::Ok) {
        RDMA_LOG(ERROR) << "Failed to execute RDMA read batch: " << res_s.desc;
        return {};
    }

    // Wait for the batched operation to complete.
    auto res_p = qp_shared->wait_one_comp();
    if (res_p != IOCode::Ok) {
        RDMA_LOG(ERROR) << "Failed to wait for RDMA completion." << std::endl;
        return {};
    }

    // Deserialize the sub-indices in parallel.
    std::atomic<bool> error_occurred(false);
#pragma omp parallel for
    for (size_t i = 0; i < batch_size; ++i) {
        uint8_t* data_ptr = recv_buffer + local_offsets[i];
        size_t data_size = sizes[i];
        std::vector<uint8_t> serialized_data(data_ptr, data_ptr + data_size);
        faiss::IndexHNSWFlat* sub_index = DistributedHnsw::deserialize_sub_hnsw(serialized_data);
        if (sub_index == nullptr) {
            RDMA_LOG(ERROR) << "Failed to deserialize sub_hnsw index for sub_idx: " << sub_indices[i];
            error_occurred.store(true, std::memory_order_relaxed);
        } else {
            sub_hnsw_list[i] = sub_index;
        }
    }
    if (error_occurred.load(std::memory_order_relaxed)) {
        return {};
    }

    // Optionally clear only the used portion of the buffer.
    memset(recv_buffer, 0, total_size);

    return sub_hnsw_list;
}

void LocalHnsw::rdma_write_to_remote(const uint8_t* data, size_t size, size_t remote_offset) {
    // Access local memory
    uint8_t* local_mem_ptr = reinterpret_cast<uint8_t*>(local_mem_->raw_ptr);
    size_t local_mem_size = local_mem_->sz;


    // Ensure that the local memory region has enough space
    if (size > local_mem_size) {
        std::cerr << "Local MR size is insufficient for RDMA write." << std::endl;
        return;
    }

    // Copy data to local memory
    memcpy(local_mem_ptr, data, size);

    // Perform RDMA write
    RMem::raw_ptr_t local_addr = reinterpret_cast<RMem::raw_ptr_t>(local_mem_ptr);
    uint64_t remote_addr = remote_attr.buf + remote_offset;

    auto res_s = qp->send_normal(
        {.op = IBV_WR_RDMA_WRITE,
         .flags = IBV_SEND_SIGNALED,
         .len = static_cast<uint32_t>(size),
         .wr_id = 0},
        {.local_addr = local_addr,
         .remote_addr = remote_addr,
         .imm_data = 0});
    RDMA_ASSERT(res_s == IOCode::Ok) << "Failed to post RDMA write: " << res_s.desc;

    // Wait for completion
    auto res_p = qp->wait_one_comp();
    RDMA_ASSERT(res_p == IOCode::Ok) << "Failed to get RDMA write completion " << std::endl;
}



// void LocalHnsw::insert_with_record_local(const int n, const std::vector<float>& data) {
//     // std::cout<<n<<std::endl;
//     float* construct_distances = new float[n];
//     dhnsw_idx_t* construct_labels = new dhnsw_idx_t[n];
//     std::vector<int> sub_hnsw_toinsert; 
//     this->meta_search(n, data.data(), 1, construct_distances, construct_labels, sub_hnsw_toinsert);
//     delete[] construct_distances;
//     delete[] construct_labels;
//     std::vector<std::vector<float>> insertset(num_sub_hnsw);// vcector:sub_hnsw unordered_set:query index 

//     for (int i = 0; i < n ; i++) {
//         int sub_idx = sub_hnsw_toinsert[i];
//         if(sub_idx >= num_sub_hnsw){
//             std::cerr << "Invalid sub-index: " << sub_idx << std::endl;
//             continue;
//         }
//         insertset[sub_idx].insert(insertset[sub_idx].end(), data.begin() + i * d, data.begin() + (i + 1) * d);
//         mapping[sub_idx].push_back(i);
//     } 

//     for (int i = 0; i < num_sub_hnsw; i++) {
//         size_t num_inserts = insertset[i].size();
//         if (!insertset[i].empty()) {
//             cache_.cache_map_[i].first->add(insertset[i].size() / d, insertset[i].data());
//         }
//         std::vector<uint8_t> serialized_sub_hnsw = serialize_sub_hnsw(i);
//         size_t sub_hnsw_size = serialized_sub_hnsw.size();
//         if( sub_hnsw_size <= offset[(i + 1) * 2 ] - offset[i * 2]){
//            //TODO: write new serialized into remote memory in memory node 
//            offset[i * 2 + 1] = offset[i * 2] + sub_hnsw_size;
//            //TODO: sychrnoize with remote node(computing and memory node)
//         }
//         else{
//             size_t new_size = 0; // calculate new size
//             int j = i;
//             while(new_size < sub_hnsw_size && j <= num_sub_hnsw){
//                 new_size += offset[j * 2 + 1] - offset[j * 2];
//                 j++;
//             }
            
//             //TODO: write new serialized into remote memory in memory node (may fail)
//             //update offset
//             offset[i * 2 + 1] = offset[i * 2] + sub_hnsw_size;
//             for(int k = i + 1; k < j; k++){
//                 //move offset
//                 offset[k * 2 + 0] = offset[k * 2 + 0] + sub_hnsw_size;
//                 offset[k * 2 + 1] = offset[k * 2 + 1] + sub_hnsw_size;
//             }
            
            
//         }
//     }
    
// }

std::vector<std::vector<dhnsw_idx_t>> LocalHnsw::get_local_mapping(){
    return this->mapping;
}


faiss::IndexHNSWFlat* LocalHnsw::fetch_sub_hnsw_for_insert(int sub_idx) {
    // Validate sub_idx
    if (sub_idx < 0 || sub_idx >= num_sub_hnsw) {
        std::cerr << "Invalid sub_idx: " << sub_idx << std::endl;
        return nullptr;
    }

    // Calculate positions and length
    uint64_t start_pos = cached_offsets_.offset_sub_hnsw[sub_idx * 2];
    uint64_t end_pos = cached_offsets_.offset_sub_hnsw[sub_idx * 2 + 1];
    u32 length = static_cast<u32>(end_pos - start_pos);

    // Access local memory
    if (!local_mem_) {
        std::cerr << "Error: local_mem_ is null" << std::endl;
        return nullptr;
    }

    if (!local_mem_->raw_ptr) {
        std::cerr << "Error: local_mem_->raw_ptr is null" << std::endl;
        return nullptr;
    }

    uint8_t* recv_buffer = reinterpret_cast<uint8_t*>(local_mem_->raw_ptr);
    size_t local_mem_size = local_mem_->sz;

    // Ensure the local buffer is large enough
    if (length > local_mem_size) {
        std::cerr << "Data length exceeds local buffer size" << std::endl;
        return nullptr;
    }

    // Perform RDMA read to fetch the data
    {
    auto res_s_1 = qp->send_normal(
        {
            .op = IBV_WR_RDMA_READ,
            .flags = IBV_SEND_SIGNALED,
            .len = (u32)length,
            .wr_id = 0
        },
        {
            .local_addr = reinterpret_cast<RMem::raw_ptr_t>(recv_buffer),
            .remote_addr = start_pos,
            .imm_data = 0
        }
    );
    RDMA_ASSERT(res_s_1 == IOCode::Ok) << "RDMA read failed: " << res_s_1.desc;

    // Wait for completion
    auto res_p_1 = qp->wait_one_comp();
    RDMA_ASSERT(res_p_1 == IOCode::Ok) << "RDMA read completion failed ";
    }

    
    size_t overflow_size = cached_offsets_.overflow[sub_idx * 52 + 1] - cached_offsets_.overflow[sub_idx * 52];
    // Perform RDMA read to fetch the overflow data
    {
    auto res_s_2 = qp->send_normal(
        {
            .op = IBV_WR_RDMA_READ,
            .flags = IBV_SEND_SIGNALED,
            .len = (u32)overflow_size,
            .wr_id = 0
        },
        {
            .local_addr = reinterpret_cast<RMem::raw_ptr_t>(recv_buffer + length),
            .remote_addr = (u64)cached_offsets_.overflow[sub_idx * 52],
            .imm_data = 0
        }
    );

    RDMA_ASSERT(res_s_2 == IOCode::Ok) << "RDMA read failed: " << res_s_2.desc;

    // Wait for completion
    auto res_p_2 = qp->wait_one_comp();
    RDMA_ASSERT(res_p_2 == IOCode::Ok) << "RDMA read completion failed ";
    }
    // Deserialize the sub-index
    std::vector<uint8_t> serialized_data(recv_buffer, recv_buffer + length);
    faiss::IndexHNSWFlat* sub_index = new faiss::IndexHNSWFlat();
    faiss::read_dhnsw_single_sub_hnsw(sub_index, serialized_data, cached_offsets_.offset_para, cached_offsets_.offset_sub_hnsw, sub_idx);

    if (!sub_index) {
        std::cerr << "Failed to deserialize sub_hnsw for sub_idx: " << sub_idx << std::endl;
        return nullptr;
    }

    // Handle potential overflows
    size_t base_idx = sub_idx * 52 + 2; 
    for(int j = 0; j < 10; j++) {
        if(cached_offsets_.overflow[base_idx + j * 5] != 0) {
            // hnsw->levels overflow
            size_t overflow_levels_start = cached_offsets_.overflow[base_idx + j * 5] - cached_offsets_.overflow[base_idx - 2] + length;
            size_t overflow_levels_end = cached_offsets_.overflow[base_idx + j * 5 + 1] - cached_offsets_.overflow[base_idx - 2] + length;
            std::vector<uint8_t> overflow_data(recv_buffer + overflow_levels_start, recv_buffer + overflow_levels_end);
            std::cout << "sub hnsw " << sub_idx << " hnsw->levels overflow " << std::endl;
            append_levels_data(sub_index, overflow_data);
        }
        else {
            break;
        }

        if(cached_offsets_.overflow[base_idx + j * 5 + 1] != 0) {
            // hnsw->offsets overflow
            size_t overflow_offsets_start = cached_offsets_.overflow[base_idx + j * 5 + 1] - cached_offsets_.overflow[base_idx - 2] + length;
            size_t overflow_offsets_end = cached_offsets_.overflow[base_idx + j * 5 + 2] - cached_offsets_.overflow[base_idx - 2] + length;
            std::vector<uint8_t> overflow_data(recv_buffer + overflow_offsets_start, recv_buffer + overflow_offsets_end);
            std::cout << "sub hnsw " << sub_idx << " hnsw->offsets overflow " << std::endl;
            append_offsets_data(sub_index, overflow_data);
        }
        else {
            break;
        }

        if(cached_offsets_.overflow[base_idx + j * 5 + 2] != 0) {
            // hnsw->neighbors overflow
            size_t overflow_neighbors_start = cached_offsets_.overflow[base_idx + j * 5 + 2] - cached_offsets_.overflow[base_idx - 2] + length;
            size_t overflow_neighbors_end = cached_offsets_.overflow[base_idx + j * 5 + 3] - cached_offsets_.overflow[base_idx - 2] + length;
            std::vector<uint8_t> overflow_data(recv_buffer + overflow_neighbors_start, recv_buffer + overflow_neighbors_end);
            std::cout << "sub hnsw " << sub_idx << " hnsw->neighbors overflow " << std::endl;
            append_neighbors_data(sub_index, overflow_data);
        }
        else {
            break;
        }

        if(cached_offsets_.overflow[base_idx + j * 5 + 3] != 0) {
            // storage->idxf->xb overflow
            size_t overflow_xb_start = cached_offsets_.overflow[base_idx + j * 5 + 3] - cached_offsets_.overflow[base_idx - 2] + length;
            size_t overflow_xb_end = cached_offsets_.overflow[base_idx + j * 5 + 4] - cached_offsets_.overflow[base_idx - 2] + length + 1;
            std::vector<uint8_t> overflow_data(recv_buffer + overflow_xb_start, recv_buffer + overflow_xb_end);
            std::cout << "sub hnsw " << sub_idx << " storage->idxf->xb overflow " << std::endl;
            append_xb_data(sub_index, overflow_data, d);
        }
        else {
            break;
        }
    }

    // Optionally clear the buffer
    memset(recv_buffer, 0, length + overflow_size);

    return sub_index;
}

std::vector<faiss::IndexHNSWFlat*> LocalHnsw::fetch_sub_hnsw_batch_for_insert(const std::vector<int>& sub_indices) {
    size_t batch_size = sub_indices.size();
    std::vector<faiss::IndexHNSWFlat*> sub_hnsw_list(batch_size, nullptr);

    // Prepare vectors for RDMA operations
    std::vector<uint64_t> sizes(batch_size);
    std::vector<uint64_t> overflow_sizes(batch_size);
    std::vector<uint64_t> local_offsets(batch_size);
    std::vector<uint64_t> remote_offsets(batch_size);
    std::vector<uint64_t> remote_overflow_offsets(batch_size);
    std::vector<uint64_t> local_overflow_offsets(batch_size);
    size_t total_size = 0;

    // Determine sizes and offsets with overflow checks
    for (size_t i = 0; i < batch_size; ++i) {
        int sub_idx = sub_indices[i];
        if (sub_idx < 0 || sub_idx >= cached_offsets_.offset_sub_hnsw.size() / 2) {
            std::cerr << "Invalid sub_idx: " << sub_idx << std::endl;
            return {};
        }
        size_t index_start = cached_offsets_.offset_sub_hnsw[sub_idx * 2];
        size_t index_end = cached_offsets_.offset_sub_hnsw[sub_idx * 2 + 1]; 
        if (index_end < index_start) {
            std::cerr << "Invalid offset range for sub_idx: " << sub_idx << std::endl;
            return {};
        }
        size_t size = index_end - index_start;

        if (total_size + size < total_size) { // Check for size overflow
            std::cerr << "Size overflow detected for sub_idx: " << sub_idx << std::endl;
            return {};
        }

        sizes[i] = size;
        remote_offsets[i] = index_start;
        local_offsets[i] = total_size;

        total_size += size;

        // overflow
        remote_overflow_offsets[i] = cached_offsets_.overflow[sub_idx * 52];
        local_overflow_offsets[i] = total_size;
        size_t overflow_size = cached_offsets_.overflow[sub_idx * 52 + 1] - cached_offsets_.overflow[sub_idx * 52];
        overflow_sizes[i] = overflow_size;
        total_size += overflow_size;
    }

    // Access local memory with null checks
    if (!local_mem_ || !local_mem_->raw_ptr) {
        std::cerr << "Error: local_mem_ or local_mem_->raw_ptr is null." << std::endl;
        return {};
    }

    uint8_t* recv_buffer = reinterpret_cast<uint8_t*>(local_mem_->raw_ptr);
    size_t local_mem_size = local_mem_->sz;

    // Ensure the local memory region has enough space
    if (total_size > local_mem_size) {
        RDMA_LOG(ERROR) << "Local MR size is insufficient for batch fetch. Required size: " << total_size;
        return {};
    }

    // Perform RDMA reads for each sub-index with additional checks
    for (size_t i = 0; i < batch_size; ++i) {
        if (local_offsets[i] + sizes[i] > local_mem_size) {
            std::cerr << "RDMA read for sub_idx " << sub_indices[i] << " exceeds local memory bounds." << std::endl;
            return {};
        }

        auto res_s = qp->send_normal(
        {
            .op = IBV_WR_RDMA_READ,
            .flags = IBV_SEND_SIGNALED,
            .len = (u32)sizes[i],
            .wr_id = 0
        },
        {
            .local_addr = reinterpret_cast<RMem::raw_ptr_t>(recv_buffer),
            .remote_addr = (u64)remote_offsets[i],
            .imm_data = 0
        }
        );

        if (res_s != IOCode::Ok) {
            RDMA_LOG(ERROR) << "Failed to post RDMA read for sub_index " << sub_indices[i] << ": " << res_s.desc;
            return {};
        }

        // Wait for completion
        auto res_p = qp->wait_one_comp();
        if (res_p != IOCode::Ok) {
            RDMA_LOG(ERROR) << "Failed to get RDMA read completion for sub_index " << sub_indices[i];
            return {};
        }

        // overflow
        auto res_s_2 = qp->send_normal(
        {
            .op = IBV_WR_RDMA_READ,
            .flags = IBV_SEND_SIGNALED,
            .len = (u32)overflow_sizes[i],
            .wr_id = 0
        },
        {
            .local_addr = reinterpret_cast<RMem::raw_ptr_t>(recv_buffer + sizes[i]),
            .remote_addr = (u64)remote_overflow_offsets[i],
            .imm_data = 0
        }
        );

        if (res_s_2 != IOCode::Ok) {
            RDMA_LOG(ERROR) << "Failed to post RDMA read for sub_index " << sub_indices[i] << ": " << res_s_2.desc;
            return {};
        }

        // Wait for completion
        auto res_p_2 = qp->wait_one_comp();
        if (res_p_2 != IOCode::Ok) {
            RDMA_LOG(ERROR) << "Failed to get RDMA read completion for sub_index " << sub_indices[i];
            return {};
        }
    }

    // Deserialize the sub-indices with size validation
    for (size_t i = 0; i < batch_size; ++i) {
        uint8_t* data_ptr = recv_buffer + local_offsets[i];
        size_t data_size = sizes[i];

        std::vector<uint8_t> serialized_data(data_ptr, data_ptr + data_size);
        faiss::IndexHNSWFlat* sub_index = new faiss::IndexHNSWFlat();
        faiss::read_dhnsw_single_sub_hnsw(sub_index, serialized_data, cached_offsets_.offset_para, cached_offsets_.offset_sub_hnsw, sub_indices[i]);

        if (!sub_index) {
            std::cerr << "Failed to deserialize sub_hnsw for sub_idx: " << sub_indices[i] << std::endl;
            delete sub_index;
            return {};
        }

        // Handle potential overflows
        size_t base_idx = sub_indices[i] * 52 + 2; 
        for(int j = 0; j < 10; j++) {
            if(cached_offsets_.overflow[base_idx + j * 5] != 0) {
                // hnsw->levels overflow
                size_t overflow_levels_start = cached_offsets_.overflow[base_idx + j * 5] - cached_offsets_.overflow[base_idx - 2] + sizes[i];
                size_t overflow_levels_end = cached_offsets_.overflow[base_idx + j * 5 + 1] - cached_offsets_.overflow[base_idx - 2] + sizes[i];
                std::vector<uint8_t> overflow_data(recv_buffer + overflow_levels_start, recv_buffer + overflow_levels_end);
                std::cout << "sub hnsw " << sub_indices[i] << " hnsw->levels overflow " << std::endl;
                append_levels_data(sub_index, overflow_data);
            }
            else {
                break;
            }

            if(cached_offsets_.overflow[base_idx + j * 5 + 1] != 0) {
                // hnsw->offsets overflow
                size_t overflow_offsets_start = cached_offsets_.overflow[base_idx + j * 5 + 1] - cached_offsets_.overflow[base_idx - 2] + sizes[i];
                size_t overflow_offsets_end = cached_offsets_.overflow[base_idx + j * 5 + 2] - cached_offsets_.overflow[base_idx - 2] + sizes[i];
                std::vector<uint8_t> overflow_data(recv_buffer + overflow_offsets_start, recv_buffer + overflow_offsets_end);
                std::cout << "sub hnsw " << sub_indices[i] << " hnsw->offsets overflow " << std::endl;
                append_offsets_data(sub_index, overflow_data);
            }
            else {
                break;
            }

            if(cached_offsets_.overflow[base_idx + j * 5 + 2] != 0) {
                // hnsw->neighbors overflow
                size_t overflow_neighbors_start = cached_offsets_.overflow[base_idx + j * 5 + 2] - cached_offsets_.overflow[base_idx - 2] + sizes[i];
                size_t overflow_neighbors_end = cached_offsets_.overflow[base_idx + j * 5 + 3] - cached_offsets_.overflow[base_idx - 2] + sizes[i];
                std::vector<uint8_t> overflow_data(recv_buffer + overflow_neighbors_start, recv_buffer + overflow_neighbors_end);
                std::cout << "sub hnsw " << sub_indices[i] << " hnsw->neighbors overflow " << std::endl;
                append_neighbors_data(sub_index, overflow_data);
            }
            else {
                break;
            }

            if(cached_offsets_.overflow[base_idx + j * 5 + 3] != 0) {
                // storage->idxf->xb overflow
                size_t overflow_xb_start = cached_offsets_.overflow[base_idx + j * 5 + 3] - cached_offsets_.overflow[base_idx - 2] + sizes[i];
                size_t overflow_xb_end = cached_offsets_.overflow[base_idx + j * 5 + 4 ] - cached_offsets_.overflow[base_idx - 2] + sizes[i] + 1;
                std::vector<uint8_t> overflow_data(recv_buffer + overflow_xb_start, recv_buffer + overflow_xb_end);
                std::cout << "sub hnsw " << sub_indices[i] << " storage->idxf->xb overflow " << std::endl;
                append_xb_data(sub_index, overflow_data, d);
            }
            else {
                break;
            }
        }

        sub_hnsw_list[i] = sub_index;
    }

    // Optionally clear the buffer
    memset(recv_buffer, 0, total_size);

    return sub_hnsw_list;
}

std::vector<faiss::IndexHNSWFlat*> LocalHnsw::fetch_sub_hnsw_batch_with_doorbell_for_insert(const std::vector<int>& sub_indices) {
    size_t batch_size = sub_indices.size();
    std::vector<faiss::IndexHNSWFlat*> sub_hnsw_list(batch_size, nullptr);

    // Prepare vectors for RDMA operations
    std::vector<uint64_t> sizes(batch_size);
    std::vector<uint64_t> local_offsets(batch_size);
    std::vector<uint64_t> remote_offsets(batch_size);
    std::vector<uint64_t> overflow_sizes(batch_size);
    std::vector<uint64_t> remote_overflow_offsets(batch_size);
    size_t total_size = 0;

    // Determine sizes and offsets with overflow checks
    for (size_t i = 0; i < batch_size; ++i) {
        int sub_idx = sub_indices[i];
        if (sub_idx < 0 || sub_idx >= cached_offsets_.offset_sub_hnsw.size() / 2) {
            std::cerr << "Invalid sub_idx: " << sub_idx << std::endl;
            return {};
        }
        size_t index_start = cached_offsets_.offset_sub_hnsw[sub_idx * 2];
        size_t index_end = cached_offsets_.offset_sub_hnsw[sub_idx * 2 + 1];
        if (index_end < index_start) {
            std::cerr << "Invalid offset range for sub_idx: " << sub_idx << std::endl;
            return {};
        }
        size_t size = index_end - index_start;

        if (total_size + size < total_size) { // Check for size overflow
            std::cerr << "Size overflow detected for sub_idx: " << sub_idx << std::endl;
            return {};
        }

        sizes[i] = size;
        remote_offsets[i] = index_start;
        local_offsets[i] = total_size;

        total_size += size;

        // Handle overflow
        if (cached_offsets_.overflow.size() > sub_idx * 52 + 7) { // Ensure indices are within bounds
            overflow_sizes[i] = cached_offsets_.overflow[sub_idx * 52 + 1] - cached_offsets_.overflow[sub_idx * 52];
            remote_overflow_offsets[i] = cached_offsets_.overflow[sub_idx * 52 + 2]; // Example offset for overflow
            total_size += overflow_sizes[i];
        }
    }

    // Access local memory with null checks
    if (!local_mem_ || !local_mem_->raw_ptr) {
        std::cerr << "Error: local_mem_ or local_mem_->raw_ptr is null." << std::endl;
        return {};
    }

    uint8_t* recv_buffer = reinterpret_cast<uint8_t*>(local_mem_->raw_ptr);
    size_t local_mem_size = local_mem_->sz;

    // Ensure the local memory region has enough space
    if (total_size > local_mem_size) {
        RDMA_LOG(ERROR) << "Local MR size is insufficient for batch fetch. Required size: " << total_size;
        return {};
    }

    const int max_batch_size = num_sub_hnsw/10;  
    if(batch_size > max_batch_size){
        std::cerr << "Batch size exceeds the maximum batch size" << std::endl;
        return {};
    }
    BenchOp<> ops[max_batch_size * 2]; // *2 to accommodate overflow operations

    u32 lkey = local_mr->get_reg_attr().value().key;
    u32 rkey = remote_attr.key;

    for (int i = 0; i < batch_size; ++i) {
        // RDMA_READ for main data
        ops[i].set_type(0);  // RDMA_READ
        ops[i].init_rbuf(reinterpret_cast<u64*>(remote_attr.buf + remote_offsets[i]), rkey);
        ops[i].init_lbuf(reinterpret_cast<u64*>(recv_buffer + local_offsets[i]), sizes[i], lkey);
        ops[i].set_wrid(i);
        ops[i].set_flags(0);  // Clear flags
        if (i != 0) {
            ops[i - 1].set_next(&ops[i]);  // Chain the operations
        }

        // RDMA_READ for overflow data
        if (overflow_sizes[i] > 0) {
            ops[batch_size + i].set_type(0);  // RDMA_READ
            ops[batch_size + i].init_rbuf(reinterpret_cast<u64*>(remote_attr.buf + remote_overflow_offsets[i]), rkey);
            ops[batch_size + i].init_lbuf(reinterpret_cast<u64*>(recv_buffer + local_offsets[i] + sizes[i]), overflow_sizes[i], lkey);
            ops[batch_size + i].set_wrid(i + batch_size);
            ops[batch_size + i].set_flags(0);  // Clear flags
            if (i != 0) {
                ops[batch_size + i - 1].set_next(&ops[batch_size + i]);  // Chain the operations
            }
        }
    }

    if (batch_size > 0 && overflow_sizes[0] > 0) {
        ops[(batch_size * 2) - 1].set_flags(IBV_SEND_SIGNALED);
    } else if (batch_size > 0) {
        ops[batch_size - 1].set_flags(IBV_SEND_SIGNALED);
    }

    // Execute the batch
    auto res_s = ops[0].execute_batch(qp_shared);
    if (res_s != IOCode::Ok) {
        RDMA_LOG(ERROR) << "Failed to execute RDMA read batch: " << res_s.desc;
        return {};
    }

    // Wait for completion
    auto res_p = qp_shared->wait_one_comp();
    if (res_p != IOCode::Ok) {
        RDMA_LOG(ERROR) << "Failed to wait for RDMA completion" << std::endl;
        return {};
    }

    // Deserialize the sub-indices with size validation
    std::atomic<bool> error_occurred(false);
     
    #pragma omp parallel for
    for (size_t i = 0; i < batch_size; ++i) {
        // Deserialize main data
        if (local_offsets[i] + sizes[i] > local_mem_size) {
            RDMA_LOG(ERROR) << "Deserialization buffer for sub_idx " << sub_indices[i] << " exceeds local memory bounds.";
            error_occurred.store(true, std::memory_order_relaxed);
            continue;
        }

        uint8_t* data_ptr = recv_buffer + local_offsets[i];
        size_t data_size = sizes[i];

        std::vector<uint8_t> serialized_data(data_ptr, data_ptr + data_size);
        faiss::IndexHNSWFlat* sub_index = new faiss::IndexHNSWFlat();
        faiss::read_dhnsw_single_sub_hnsw(sub_index, serialized_data, cached_offsets_.offset_para, cached_offsets_.offset_sub_hnsw, sub_indices[i]);

        if (sub_index == nullptr) {
            RDMA_LOG(ERROR) << "Failed to deserialize sub_hnsw index for sub_idx: " << sub_indices[i];
            error_occurred.store(true, std::memory_order_relaxed);
            continue;
        }


        // Handle potential overflows
        size_t base_idx = sub_indices[i] * 52 + 2; 
        for(int j = 0; j < 10; j++) {
            if(cached_offsets_.overflow[base_idx + j * 8] != 0) {
                // hnsw->levels overflow
                size_t overflow_levels_start = cached_offsets_.overflow[base_idx + j * 8] - cached_offsets_.overflow[base_idx - 2] + sizes[i];
                size_t overflow_levels_end = cached_offsets_.overflow[base_idx + j * 8 + 1] - cached_offsets_.overflow[base_idx - 2] + sizes[i];
                std::vector<uint8_t> overflow_data(recv_buffer + overflow_levels_start, recv_buffer + overflow_levels_end);
                std::cout << "sub hnsw " << sub_indices[i] << " hnsw->levels overflow " << std::endl;
                append_levels_data(sub_index, overflow_data);
            }
            else {
                break;
            }

            if(cached_offsets_.overflow[base_idx + j * 8 + 2] != 0) {
                // hnsw->offsets overflow
                size_t overflow_offsets_start = cached_offsets_.overflow[base_idx + j * 8 + 2] - cached_offsets_.overflow[base_idx - 2] + sizes[i];
                size_t overflow_offsets_end = cached_offsets_.overflow[base_idx + j * 8 + 3] - cached_offsets_.overflow[base_idx - 2] + sizes[i];
                std::vector<uint8_t> overflow_data(recv_buffer + overflow_offsets_start, recv_buffer + overflow_offsets_end);
                std::cout << "sub hnsw " << sub_indices[i] << " hnsw->offsets overflow " << std::endl;
                append_offsets_data(sub_index, overflow_data);
            }
            else {
                break;
            }

            if(cached_offsets_.overflow[base_idx + j * 8 + 4] != 0) {
                // hnsw->neighbors overflow
                size_t overflow_neighbors_start = cached_offsets_.overflow[base_idx + j * 8 + 4] - cached_offsets_.overflow[base_idx - 2] + sizes[i];
                size_t overflow_neighbors_end = cached_offsets_.overflow[base_idx + j * 8 + 5] - cached_offsets_.overflow[base_idx - 2] + sizes[i];
                std::vector<uint8_t> overflow_data(recv_buffer + overflow_neighbors_start, recv_buffer + overflow_neighbors_end);
                std::cout << "sub hnsw " << sub_indices[i] << " hnsw->neighbors overflow " << std::endl;
                append_neighbors_data(sub_index, overflow_data);
            }
            else {
                break;
            }

            if(cached_offsets_.overflow[base_idx + j * 8 + 6] != 0) {
                // storage->idxf->xb overflow
                size_t overflow_xb_start = cached_offsets_.overflow[base_idx + j * 8 + 6] - cached_offsets_.overflow[base_idx - 2] + sizes[i];
                size_t overflow_xb_end = cached_offsets_.overflow[base_idx + j * 8 + 7] - cached_offsets_.overflow[base_idx - 2] + sizes[i];
                std::vector<uint8_t> overflow_data(recv_buffer + overflow_xb_start, recv_buffer + overflow_xb_end);
                std::cout << "sub hnsw " << sub_indices[i] << " storage->idxf->xb overflow " << std::endl;
                append_xb_data(sub_index, overflow_data, d);
            }
            else {
                break;
            }
        }

        sub_hnsw_list[i] = sub_index;
    }

    if (error_occurred.load(std::memory_order_relaxed)) {
        return {};
    } 

    // Clear the buffer safely
    memset(recv_buffer, 0, total_size);
    return sub_hnsw_list;
}

// void LocalHnsw::insert_origin(const int n, const std::vector<float>& data, fetch_type flag){
//     float* meta_distances = new float[n];
//     dhnsw_idx_t* meta_labels = new dhnsw_idx_t[n];
//     std::vector<int> sub_hnsw_toinsert;
//     this->meta_search(n, data.data(), 1, meta_distances, meta_labels, sub_hnsw_toinsert);
//     delete[] meta_distances;
//     delete[] meta_labels;
//     this->insert_rdma(n, data, flag, sub_hnsw_toinsert);
//     //TODO: update
// }

// void LocalHnsw::insert_rdma(const int n, const std::vector<float>& data, fetch_type flag, std::vector<int>& sub_hnsw_toinsert) {
//     // Separate sub_hnsw indices into cached and uncached
//     std::vector<int> cached_sub_indices;
//     std::vector<int> uncached_sub_indices;
//     std::unordered_map<int, std::unordered_set<int>> cached_insertset;
//     std::unordered_map<int, std::unordered_set<int>> uncached_insertset;

//     for (int i = 0; i < n; i++) {
//         int sub_idx = sub_hnsw_toinsert[i];        
//         // Check if the sub_hnsw is in the cache
//         if (cache_.get(sub_idx) != nullptr) {
//             // Sub_hnsw is in cache
//             cached_sub_indices.push_back(sub_idx);
//             cached_insertset[sub_idx].insert(i);
//         } else {
//             // Sub_hnsw not in cache
//             uncached_sub_indices.push_back(sub_idx);
//             uncached_insertset[sub_idx].insert(i);
//         }
//     }

//     // First process cached sub_hnsw indices
//     for (int sub_idx : cached_sub_indices) {
//         const std::unordered_set<int>& insert_indices_set = cached_insertset[sub_idx];

//         // Sub_hnsw is in cache
//         faiss::IndexHNSWFlat* sub_index = cache_.get(sub_idx);

//         // Proceed with insert on sub_index
//         size_t num_insert = insert_indices_set.size();
//         std::vector<float> tmp_data(num_insert * d);
//         for (auto it = insert_indices_set.begin(); it != insert_indices_set.end(); ++it) {
//             int insert_idx = *it;
//             std::copy(data.begin() + insert_idx * d, data.begin() + (insert_idx + 1) * d, tmp_data.end());
//         }
//         InsertionContext ctx = {
//                     .sub_idx = sub_idx,
//                     .n = (size_t)sub_index->ntotal,
//                     .data = tmp_data.data(),
//                     .entry_point = sub_index->hnsw.entry_point,
//                     .max_level = sub_index->hnsw.max_level,
//                 };       

//         int current_max_level = sub_index->hnsw.max_level;
//         // Force new points to be inserted at level 0 or 1 only
//         sub_index->hnsw.max_level = std::min(1, current_max_level);

//         sub_index->add(num_insert, tmp_data.data());

//         // Restore original max_level
//         sub_index->hnsw.max_level = current_max_level;
//         update_remote_data_with_doorbell(ctx);
//     }

//     // Then process uncached sub_hnsw indices in batches
//     size_t batch_size = num_sub_hnsw/10; // Maximum number of sub_hnsw to fetch together
//     size_t num_uncached = uncached_sub_indices.size();
//     for (size_t i = 0; i < num_uncached; i += batch_size) {
//         size_t current_batch_size = std::min(batch_size, num_uncached - i);
//         std::vector<int> batch_sub_indices(uncached_sub_indices.begin() + i, uncached_sub_indices.begin() + i + current_batch_size);
//         std::vector<faiss::IndexHNSWFlat*> sub_indices;
//         if(flag == RDMA_DOORBELL){
//             // Fetch the batch of sub_hnsw from the server using RDMA doorbell
//             sub_indices = fetch_sub_hnsw_batch_with_doorbell_for_insert(batch_sub_indices);
//         }
//         else if(flag == RDMA){
//             // Fetch the batch of sub_hnsw from the server using RDMA
//             // std::cout << "Fetching sub_hnsw using RDMA" << std::endl;
//             sub_indices = fetch_sub_hnsw_batch_for_insert(batch_sub_indices);
            
//         }
//         else{
//             std::cerr << "Invalid flag" << std::endl;
//             return;
//         }
//         // std::cout << "Fetched " << sub_indices.size() << " sub_hnsw" << std::endl;
//         // Process each fetched sub_hnsw
//         for (size_t idx = 0; idx < batch_sub_indices.size(); ++idx) {
//             int sub_idx = batch_sub_indices[idx];
//             faiss::IndexHNSWFlat* sub_index = sub_indices[idx];

//             cache_.put(sub_idx, sub_index);

//             // Proceed with search on sub_index
//             const std::unordered_set<int>& insert_indices_set = uncached_insertset[sub_idx];
            
//             size_t num_insert = insert_indices_set.size();

//             std::vector<float> tmp_data(num_insert * d);

//             for (auto it = insert_indices_set.begin(); it != insert_indices_set.end(); ++it) {
//                 int insert_idx = *it;
//                 std::copy(data.begin() + insert_idx * d, data.begin() + (insert_idx + 1) * d, tmp_data.end());
//             }

//             InsertionContext ctx = {
//                     .sub_idx = sub_idx,
//                     .n = (size_t)sub_index->ntotal,
//                     .data = tmp_data.data(),
//                     .entry_point = sub_index->hnsw.entry_point,
//                     .max_level = sub_index->hnsw.max_level,
//             };

//             int current_max_level = sub_index->hnsw.max_level;
//             // Force new points to be inserted at level 0 or 1 only
//             sub_index->hnsw.max_level = std::min(1, current_max_level);

//             sub_index->add(num_insert, tmp_data.data());

//             // Restore original max_level
//             sub_index->hnsw.max_level = current_max_level;

//             update_remote_data_with_doorbell(ctx);
//             //update cached_offsets_ is done in update_remote_data_with_doorbell
//             }
//         }

// } 


// void LocalHnsw::update_remote_data_with_doorbell(InsertionContext& ctx) { 
//     const int MAX_DOORBELL_OPS = 16;
//     int num_doorbell_ops = 0;
//     BenchOp<> ops[MAX_DOORBELL_OPS];


//     std::lock_guard<std::mutex> lock(update_mutex_);
    
//     size_t base_sub_idx = ctx.sub_idx * 15;
//     size_t base_overflow_idx = ctx.sub_idx * 3;
//     faiss::IndexHNSWFlat* sub_index = cache_.get(ctx.sub_idx);

//     std::vector<RDMAField> fields_to_update;
//     CachedOffsets updated_cached_offsets; // first copy cached_offsets_
//     updated_cached_offsets.overflow = cached_offsets_.overflow;
//     updated_cached_offsets.offset_para = cached_offsets_.offset_para;
//     updated_cached_offsets.offset_sub_hnsw= cached_offsets_.offset_sub_hnsw;
//     updated_cached_offsets.is_initialized = cached_offsets_.is_initialized;
//     // 1. Update ntotal fields if changed
//     if (ctx.n != sub_index->ntotal) {
//         std::vector<uint8_t> ntotal_data(sizeof(faiss::Index::idx_t));
//         std::memcpy(ntotal_data.data(), &sub_index->ntotal, sizeof(faiss::Index::idx_t));
        
//         fields_to_update.push_back({
//             .local_data = ntotal_data,
//             .remote_addr = cached_offsets_.offset_para[base_sub_idx + 0],
//             .size = sizeof(faiss::Index::idx_t)
//         });
        
//         fields_to_update.push_back({
//             .local_data = ntotal_data,
//             .remote_addr = cached_offsets_.offset_para[base_sub_idx + 7],
//             .size = sizeof(faiss::Index::idx_t)
//         });
//     }

//     // 2. Update entry_point if changed
//     if (ctx.entry_point != sub_index->hnsw.entry_point) {
//         std::vector<uint8_t> entry_point_data(sizeof(faiss::IndexHNSWFlat::storage_idx_t));
//         std::memcpy(entry_point_data.data(), &sub_index->hnsw.entry_point, 
//                     sizeof(faiss::IndexHNSWFlat::storage_idx_t));
        
//         fields_to_update.push_back({
//             .local_data = entry_point_data,
//             .remote_addr = cached_offsets_.offset_para[base_sub_idx + 5],
//             .size = sizeof(faiss::IndexHNSWFlat::storage_idx_t)
//         });
//     }

//     // 3. Update max_level if changed
//     if (ctx.max_level != sub_index->hnsw.max_level) {
//         std::vector<uint8_t> max_level_data(sizeof(int));
//         std::memcpy(max_level_data.data(), &sub_index->hnsw.max_level, sizeof(int));
        
//         fields_to_update.push_back({
//             .local_data = max_level_data,
//             .remote_addr = cached_offsets_.offset_para[base_sub_idx + 6],
//             .size = sizeof(int)
//         });
//     }

//     // 4. Update data structures if new data was inserted
//     if (ctx.data != nullptr && ctx.n < sub_index->ntotal) {
//         size_t num_insert = sub_index->ntotal - ctx.n;
//         size_t current_overflow_pos = 0;
//         size_t base_overflow_idx = ctx.sub_idx * 3;
//         size_t max_overflow_size = 0; 
//         size_t pushback_overflow_start = 0;

//         // Update xb data
//         if(ctx.sub_idx % 2 == 1 ){
//             max_overflow_size = updated_cached_offsets.overflow[base_overflow_idx + 1] - updated_cached_offsets.overflow[(ctx.sub_idx - 1) * 3 + 1];
//         }
//         else if (ctx.sub_idx != num_sub_hnsw - 1){
//             max_overflow_size = updated_cached_offsets.overflow[(ctx.sub_idx + 1) * 3 + 1] - updated_cached_offsets.overflow[base_overflow_idx + 1];
//             pushback_overflow_start = updated_cached_offsets.overflow[(ctx.sub_idx + 1) * 3 + 1]; 
//         }
//         else{
//             max_overflow_size = updated_cached_offsets.overflow[base_overflow_idx + 2] - updated_cached_offsets.overflow[base_overflow_idx + 1];
//             pushback_overflow_start = updated_cached_offsets.overflow[(ctx.sub_idx + 1) * 3 + 1]; 
//         }
        
//         size_t xb_size = num_insert * sub_index->d * sizeof(float);
//         size_t xb_end = updated_cached_offsets.offset_para[base_sub_idx + 8] + sizeof(size_t) + ctx.n * sub_index->d * sizeof(float);
//         size_t xb_max_size = updated_cached_offsets.offset_para[base_sub_idx + 12] - xb_end; //TODO: check if this is correct
        
//         if (xb_size > xb_max_size) {
//             // Split data between original space and overflow
//             std::vector<uint8_t> original_xb_data(xb_max_size); 
//             std::memcpy(original_xb_data.data(), ctx.data, xb_max_size);
            
//             size_t overflow_size = xb_size - xb_max_size;
            
//             if(overflow_size > max_overflow_size){
//                 //error
//                 std::cerr << "Overflow size is greater than max_overflow_size" << std::endl;
//             }
//             if(ctx.sub_idx % 2 == 1){
//                 pushback_overflow_start =  updated_cached_offsets.overflow[base_overflow_idx + 1] - overflow_size; 
//             }
            
//             std::vector<uint8_t> overflow_xb_data(overflow_size);
//             std::memcpy(overflow_xb_data.data(), 
//                        reinterpret_cast<const uint8_t*>(ctx.data) + xb_max_size, 
//                        overflow_size);
//             // Write to size of xb (for WRITEVECTOR) TODO: modify both data and remote_addr for all vectors fields
//             std::vector<uint8_t> xb_max_size_data(sizeof(size_t));
//             std::memcpy(xb_max_size_data.data(), &xb_max_size, sizeof(size_t));
        
//             fields_to_update.push_back({
//                 .local_data = xb_max_size_data,
//                 .remote_addr = updated_cached_offsets.offset_para[base_sub_idx + 8],
//                 .size = sizeof(size_t)
//             });

//             // Write to original space
//             fields_to_update.push_back({
//                 .local_data = original_xb_data,
//                 .remote_addr = updated_cached_offsets.offset_para[base_sub_idx + 8],
//                 .size = xb_max_size
//             });
            
//             // Write to overflow area for xb
//             fields_to_update.push_back({
//                 .local_data = overflow_xb_data,
//                 .remote_addr = pushback_overflow_start,
//                 .size = overflow_size
//             });

//             if(ctx.sub_idx % 2 == 0){
//                 updated_cached_offsets.overflow[base_overflow_idx + 1] = pushback_overflow_start + overflow_size;
//             }
//             else{
//                 updated_cached_offsets.overflow[base_overflow_idx + 1] = pushback_overflow_start;
//             }
            
//         } else {
//             std::vector<uint8_t> xb_data(xb_size);
//             std::memcpy(xb_data.data(), ctx.data, xb_size);
//             std::vector<uint8_t> xb_size_data(sizeof(size_t));
//             std::memcpy(xb_size_data.data(), &xb_size, sizeof(size_t));
//             fields_to_update.push_back({
//                 .local_data = xb_size_data,
//                 .remote_addr = updated_cached_offsets.offset_para[base_sub_idx + 8],
//                 .size = sizeof(size_t)
//             });
//             fields_to_update.push_back({
//                 .local_data = xb_data,
//                 .remote_addr = updated_cached_offsets.offset_para[base_sub_idx + 8],
//                 .size = xb_size
//             });
//         }

//         // Update levels
//         if(ctx.sub_idx % 2 == 1 ){
//             max_overflow_size = updated_cached_offsets.overflow[base_overflow_idx + 1] - updated_cached_offsets.overflow[(ctx.sub_idx - 1) * 3 + 1];
//         }
//         else if (ctx.sub_idx != num_sub_hnsw - 1){
//             max_overflow_size = updated_cached_offsets.overflow[(ctx.sub_idx + 1) * 3 + 1] - updated_cached_offsets.overflow[base_overflow_idx + 1];
//             pushback_overflow_start = updated_cached_offsets.overflow[(ctx.sub_idx + 1) * 3 + 1]; 
//         }
//         else{
//             max_overflow_size = updated_cached_offsets.overflow[base_overflow_idx + 2] - updated_cached_offsets.overflow[base_overflow_idx + 1];
//             pushback_overflow_start = updated_cached_offsets.overflow[(ctx.sub_idx + 1) * 3 + 1]; 
//         }

//         size_t total_levels = sub_index->hnsw.levels.size() * sizeof(int);
//         size_t start_level = ctx.n;
//         size_t new_levels_count = num_insert;

//         std::vector<uint8_t> levels_data(new_levels_count * sizeof(int));
//         std::memcpy(levels_data.data(), sub_index->hnsw.levels.data() + start_level, new_levels_count * sizeof(int));

//         size_t levels_size = levels_data.size();
//         size_t levels_end = updated_cached_offsets.offset_para[base_sub_idx + 1] + sizeof(size_t) + ctx.n * sizeof(int);
//         size_t levels_max_size = updated_cached_offsets.offset_para[base_sub_idx + 2] - levels_end;

//         if (levels_size > levels_max_size) {
//             std::vector<uint8_t> original_levels_data(levels_max_size);
//             std::memcpy(original_levels_data.data(), sub_index->hnsw.levels.data(), levels_max_size);
            
//             size_t overflow_size = levels_size - levels_max_size;

//             if(overflow_size > max_overflow_size){
//                 //error
//                 std::cerr << "Overflow size is greater than max_overflow_size" << std::endl;
//             }
//             if(ctx.sub_idx % 2 == 1){
//                 pushback_overflow_start =  updated_cached_offsets.overflow[base_overflow_idx + 1] - overflow_size; 
//             }

//             std::vector<uint8_t> overflow_levels_data(overflow_size);
//             std::memcpy(overflow_levels_data.data(), 
//                        reinterpret_cast<const uint8_t*>(sub_index->hnsw.levels.data()) + levels_max_size,
//                        overflow_size);
            
//             // Write to size of levels (for WRITEVECTOR) TODO: modify both data and remote_addr for all vectors fields
//             std::vector<uint8_t> levels_max_size_data(sizeof(size_t));
//             std::memcpy(levels_max_size_data.data(), &levels_max_size, sizeof(size_t));

//             fields_to_update.push_back({
//                 .local_data = levels_max_size_data,
//                 .remote_addr = updated_cached_offsets.offset_para[base_sub_idx + 1],
//                 .size = sizeof(size_t)
//             });

//             fields_to_update.push_back({
//                 .local_data = original_levels_data,
//                 .remote_addr = updated_cached_offsets.offset_para[base_sub_idx + 1],
//                 .size = levels_max_size
//             });
            
//             fields_to_update.push_back({
//                 .local_data = overflow_levels_data,
//                 .remote_addr = pushback_overflow_start,
//                 .size = overflow_size
//             });
            
//             if(ctx.sub_idx % 2 == 0){
//                 updated_cached_offsets.overflow[base_overflow_idx + 1] = pushback_overflow_start + overflow_size;
//             }
//             else{
//                 updated_cached_offsets.overflow[base_overflow_idx + 1] = pushback_overflow_start;
//             } 

//         } else {
//             std::vector<uint8_t> levels_data(levels_size);
//             std::memcpy(levels_data.data(), sub_index->hnsw.levels.data(), levels_size);
//             std::vector<uint8_t> levels_size_data(sizeof(size_t));  
//             std::memcpy(levels_size_data.data(), &levels_size, sizeof(size_t));
//             fields_to_update.push_back({
//                 .local_data = levels_size_data,
//                 .remote_addr = updated_cached_offsets.offset_para[base_sub_idx + 1],
//                 .size = sizeof(size_t)
//             });

//             fields_to_update.push_back({
//                 .local_data = levels_data,
//                 .remote_addr = updated_cached_offsets.offset_para[base_sub_idx + 1],
//                 .size = levels_size
//             });
//         }

//         // Update offsets
//         if(ctx.sub_idx % 2 == 1 ){
//             max_overflow_size = updated_cached_offsets.overflow[base_overflow_idx + 1] - updated_cached_offsets.overflow[(ctx.sub_idx - 1) * 3 + 1];
//         }
//         else if (ctx.sub_idx != num_sub_hnsw - 1){
//             max_overflow_size = updated_cached_offsets.overflow[(ctx.sub_idx + 1) * 3 + 1] - updated_cached_offsets.overflow[base_overflow_idx + 1];
//             pushback_overflow_start = updated_cached_offsets.overflow[(ctx.sub_idx + 1) * 3 + 1]; 
//         }
//         else{
//             max_overflow_size = updated_cached_offsets.overflow[base_overflow_idx + 2] - updated_cached_offsets.overflow[base_overflow_idx + 1];
//             pushback_overflow_start = updated_cached_offsets.overflow[(ctx.sub_idx + 1) * 3 + 1]; 
//         }

//         size_t total_offsets = sub_index->hnsw.offsets.size() * sizeof(size_t);
//         size_t start_offset = ctx.n + 1;
//         size_t new_offsets_count = num_insert;

//         std::vector<uint8_t> offsets_data(new_offsets_count * sizeof(size_t));
//         std::memcpy(offsets_data.data(), sub_index->hnsw.offsets.data() + start_offset, new_offsets_count * sizeof(size_t));

//         size_t offsets_size = offsets_data.size();
//         size_t offsets_end = cached_offsets_.offset_para[base_sub_idx + 2] + sizeof(size_t) + (ctx.n + 1) * sizeof(size_t);
//         size_t offsets_max_size = cached_offsets_.offset_para[base_sub_idx + 3] - offsets_end;
        
//         if (offsets_size > offsets_max_size) {
//             std::vector<uint8_t> original_offsets_data(offsets_max_size);
//             std::memcpy(original_offsets_data.data(), sub_index->hnsw.offsets.data(), offsets_max_size);
            
//             size_t overflow_size = offsets_size - offsets_max_size;

//             if(overflow_size > max_overflow_size){
//                 //error
//                 std::cerr << "Overflow size is greater than max_overflow_size" << std::endl;
//             }
//             if(ctx.sub_idx % 2 == 1){
//                 pushback_overflow_start =  updated_cached_offsets.overflow[base_overflow_idx + 1] - overflow_size; 
//             }

//             std::vector<uint8_t> overflow_offsets_data(overflow_size);
//             std::memcpy(overflow_offsets_data.data(), 
//                        reinterpret_cast<const uint8_t*>(sub_index->hnsw.offsets.data()) + offsets_max_size,
//                        overflow_size);

//             // Write to size of offsets (for WRITEVECTOR) TODO: modify both data and remote_addr for all vectors fields
//             std::vector<uint8_t> offsets_max_size_data(sizeof(size_t));
//             std::memcpy(offsets_max_size_data.data(), &offsets_max_size, sizeof(size_t));
 
//             fields_to_update.push_back({
//                 .local_data = offsets_max_size_data,
//                 .remote_addr = updated_cached_offsets.offset_para[base_sub_idx + 1],
//                 .size = sizeof(size_t)
//             });

//             fields_to_update.push_back({
//                 .local_data = original_offsets_data,
//                 .remote_addr = updated_cached_offsets.offset_para[base_sub_idx + 2],
//                 .size = offsets_max_size
//             });
            
//             fields_to_update.push_back({
//                 .local_data = overflow_offsets_data,
//                 .remote_addr = pushback_overflow_start,
//                 .size = overflow_size
//             });

//             if(ctx.sub_idx % 2 == 0){
//                 updated_cached_offsets.overflow[base_overflow_idx + 1] = pushback_overflow_start + overflow_size;
//             }
//             else{
//                 updated_cached_offsets.overflow[base_overflow_idx + 1] = pushback_overflow_start;
//             } 

//         } else {
//             std::vector<uint8_t> offsets_data(offsets_size);
//             std::memcpy(offsets_data.data(), sub_index->hnsw.offsets.data(), offsets_size);
//             std::vector<uint8_t> offsets_size_data(sizeof(size_t));     
//             std::memcpy(offsets_size_data.data(), &offsets_size, sizeof(size_t));
//             fields_to_update.push_back({
//                 .local_data = offsets_size_data,
//                 .remote_addr = updated_cached_offsets.offset_para[base_sub_idx + 2],
//                 .size = sizeof(size_t)
//             });
//             fields_to_update.push_back({
//                 .local_data = offsets_data,
//                 .remote_addr = updated_cached_offsets.offset_para[base_sub_idx + 2],
//                 .size = offsets_size
//             });
//         }

//         // Update neighbors
//          if(ctx.sub_idx % 2 == 1 ){
//             max_overflow_size = updated_cached_offsets.overflow[base_overflow_idx + 1] - updated_cached_offsets.overflow[(ctx.sub_idx - 1) * 3 + 1];
//         }
//         else if (ctx.sub_idx != num_sub_hnsw - 1){
//             max_overflow_size = updated_cached_offsets.overflow[(ctx.sub_idx + 1) * 3 + 1] - updated_cached_offsets.overflow[base_overflow_idx + 1];
//             pushback_overflow_start = updated_cached_offsets.overflow[(ctx.sub_idx + 1) * 3 + 1]; 
//         }
//         else{
//             max_overflow_size = updated_cached_offsets.overflow[base_overflow_idx + 2] - updated_cached_offsets.overflow[base_overflow_idx + 1];
//             pushback_overflow_start = updated_cached_offsets.overflow[(ctx.sub_idx + 1) * 3 + 1]; 
//         }
//         // Update neighbors in transposed format

//         // Calculate dimensions for bottom layers
//         size_t total_nodes = sub_index->hnsw.levels.size();
//         size_t num_levels = sub_index->hnsw.cum_nneighbor_per_level.size() - 1;
        
//         // Get positions of bottom two layers
//         size_t bottom_start = updated_cached_offsets.offset_para[base_sub_idx + 13];
//         size_t bottom_end = updated_cached_offsets.offset_para[base_sub_idx + 14];
//         size_t bottom_size = bottom_end - bottom_start;

//         // Prepare bottom layers data in transposed format
//         std::vector<uint8_t> bottom_neighbors;

//         // Write bottom two layers in transposed format
//         for (int level = sub_index->hnsw.max_level; level >= 0; level--) {
//             int nbrs = sub_index->hnsw.nb_neighbors(level);
//             for (int pos = 0; pos < nbrs; pos++) {
//                 for (int node = 0; node < static_cast<int>(total_nodes); node++) {
//                     if(level < sub_index->hnsw.levels[node]) {
//                     int nbrs = sub_index->hnsw.nb_neighbors(level);
//                         for (int pos = 0; pos < nbrs; pos++) { 
//                             size_t offset_start = sub_index->hnsw.offsets[node];
//                             size_t cum_neighbors_before = sub_index->hnsw.cum_nb_neighbors(level);
//                             size_t idx = offset_start + cum_neighbors_before + pos;
//                             if (idx >= sub_index->hnsw.neighbors.size()) {
//                                 throw std::runtime_error("Neighbor index out of bounds");
//                             }
//                             storage_idx_t neighbor;
//                             std::memcpy(&neighbor, reinterpret_cast<const uint8_t*>(sub_index->hnsw.neighbors.data()) + idx * sizeof(storage_idx_t), sizeof(storage_idx_t));
//                             bottom_neighbors.insert(bottom_neighbors.end(),
//                                 reinterpret_cast<const uint8_t*>(&neighbor),
//                                 reinterpret_cast<const uint8_t*>(&neighbor) + sizeof(neighbor));
//                         }
//                     }
//                 }
//                 }
//         }

//         // Check if we need overflow
//         size_t neighbors_max_size = updated_cached_offsets.offset_para[base_sub_idx + 5] - updated_cached_offsets.offset_para[base_sub_idx + 13];
//         if (bottom_neighbors.size() > neighbors_max_size) {
//             // Handle overflow
//             size_t original_size = neighbors_max_size;
//             size_t overflow_size = bottom_neighbors.size() - original_size;
            
//             // Update overflow offsets
//             if(overflow_size > max_overflow_size){
//                 //error
//                 std::cerr << "Overflow size is greater than max_overflow_size" << std::endl;
//             }
//             if(ctx.sub_idx % 2 == 1){
//                 pushback_overflow_start =  updated_cached_offsets.overflow[base_overflow_idx + 1] - overflow_size; 
//             } 

//             // Split data between original and overflow regions
//             std::vector<uint8_t> original_data(bottom_neighbors.begin(), bottom_neighbors.begin() + original_size);
//             std::vector<uint8_t> overflow_data(bottom_neighbors.begin() + original_size, bottom_neighbors.end());
            
//             // TODO: reverse level order for neighbors && check   static void read_dhnsw_HNSW in dhnsw_io.cpp
//             // Write original data
//             std::vector<uint8_t> original_size_data(sizeof(size_t));
//             std::memcpy(original_size_data.data(), &original_size, sizeof(size_t));
//             fields_to_update.push_back({
//                 .local_data = original_size_data,
//                 .remote_addr = updated_cached_offsets.offset_para[base_sub_idx + 13],
//                 .size = sizeof(size_t)
//             });

//             fields_to_update.push_back({
//                 .local_data = original_data,
//                 .remote_addr = updated_cached_offsets.offset_para[base_sub_idx + 13],
//                 .size = original_size
//             });
            
//             // Write overflow data
//             fields_to_update.push_back({
//                 .local_data = overflow_data,
//                 .remote_addr = pushback_overflow_start,
//                 .size = overflow_size
//             });

//             if(ctx.sub_idx % 2 == 0){
//                 updated_cached_offsets.overflow[base_overflow_idx + 1] = pushback_overflow_start + overflow_size;
//             }
//             else{
//                 updated_cached_offsets.overflow[base_overflow_idx + 1] = pushback_overflow_start;
//             }

//             updated_cached_offsets.offset_para[base_sub_idx + 14] = updated_cached_offsets.offset_para[base_sub_idx + 13] + original_size;
//         } else {
//             // Write entire bottom layers data
//             size_t size_val = bottom_neighbors.size();
//             std::vector<uint8_t> size_data(sizeof(size_t));
//             std::memcpy(size_data.data(), &size_val, sizeof(size_t));
            
//             fields_to_update.push_back(RDMAField{
//                 .local_data = size_data,
//                 .remote_addr = updated_cached_offsets.offset_para[base_sub_idx + 13],
//                 .size = sizeof(size_t)
//             });
//             fields_to_update.push_back(RDMAField{
//                 .local_data = bottom_neighbors,
//                 .remote_addr = updated_cached_offsets.offset_para[base_sub_idx + 13],
//                 .size = bottom_neighbors.size()
//             });
//             updated_cached_offsets.offset_para[base_sub_idx + 14] = updated_cached_offsets.offset_para[base_sub_idx + 13] + bottom_neighbors.size();
//         }
//     }

//     // Execute RDMA operations in batches
//     for (const auto& field : fields_to_update) {
//         if (num_doorbell_ops >= MAX_DOORBELL_OPS) {
//             auto res_s = ops[0].execute_batch(qp_shared);
//             if (res_s != IOCode::Ok) {
//                 RDMA_LOG(ERROR) << "Failed to execute RDMA write batch";
//                 return;
//             }
//             num_doorbell_ops = 0;
//         }

//         std::memcpy(local_mem_->raw_ptr, field.local_data.data(), field.size);
        
//         ops[num_doorbell_ops].set_type(IBV_WR_RDMA_WRITE);
//         ops[num_doorbell_ops].init_lbuf(
//             reinterpret_cast<rdmaio::u64*>(local_mem_->raw_ptr),
//             local_mr->get_reg_attr().value().key,
//             field.size
//         );
//         ops[num_doorbell_ops].init_rbuf(
//             reinterpret_cast<rdmaio::u64*>(field.remote_addr),
//             remote_attr.key,
//             field.size
//         );
//         ops[num_doorbell_ops].set_wrid(num_doorbell_ops);
//         ops[num_doorbell_ops].set_flags(IBV_SEND_SIGNALED);
        
//         num_doorbell_ops++;
//     }

//     // Execute any remaining operations
//     if (num_doorbell_ops > 0) {
//         auto res_s = ops[0].execute_batch(qp_shared);
//         if (res_s != IOCode::Ok) {
//             RDMA_LOG(ERROR) << "Failed to execute final RDMA write batch";
//         }
//     }
//     //update cached_offsets after updating remote data successfully
//     cached_offsets_.offset_para = updated_cached_offsets.offset_para;
//     cached_offsets_.overflow = updated_cached_offsets.overflow; 
// }

// ---------------------------- pipelined search ----------------------------
static std::vector<double> build_zipf_cdf(int N, double s) {
    std::vector<double> w(N);
    for (int i = 0; i < N; ++i) {            // rank = i+1
        w[i] = 1.0 / std::pow(double(i+1), s);
    }
    double sum = std::accumulate(w.begin(), w.end(), 0.0);
    for (auto &x : w) x /= sum;

    std::vector<double> cdf(N);
    std::partial_sum(w.begin(), w.end(), cdf.begin());

    cdf.back() = 1.0;
    return cdf;
}


static inline int sample_from_cdf(const std::vector<double>& cdf, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> U(0.0, 1.0);
    double u = U(rng);
    auto it = std::upper_bound(cdf.begin(), cdf.end(), u);
    int idx = int(it - cdf.begin());
    if (idx >= (int)cdf.size()) idx = (int)cdf.size() - 1;
    return idx;
}

std::unordered_map<int, std::unordered_set<int>>
LocalHnsw::meta_search_pipelined_micro(const int n,const float* query, int K_meta,
                                 float* distances, dhnsw_idx_t* labels,
                                 std::vector<int>& sub_hnsw_tosearch,
                                 int core_start, int cores_per_worker)
{
    const bool use_zipf = true;      
    const double zipf_s = 1.1;      
    const uint64_t zipf_seed = 42;   

    omp_set_num_threads(cores_per_worker);

    sub_hnsw_tosearch.clear();
    sub_hnsw_tosearch.resize(n * K_meta, -1);

    if (!use_zipf) {

        meta_hnsw->search(n, query, K_meta, distances, labels);
        for (int i = 0; i < K_meta * n; ++i) {
            int label = labels[i];
            sub_hnsw_tosearch[i] = label;
        }
    } else {
  
        if (K_meta > num_sub_hnsw) {
            fprintf(stderr, "[ERROR] K_meta(%d) > num_sub_hnsw(%d)\n", K_meta, num_sub_hnsw);
            K_meta = num_sub_hnsw; 
        }
        auto cdf = build_zipf_cdf(num_sub_hnsw, zipf_s);


        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            std::mt19937_64 rng(zipf_seed ^ (0x9E3779B97F4A7C15ULL * (tid + 1)));

            #pragma omp for schedule(static)
            for (int qi = 0; qi < n; ++qi) {
          
                std::unordered_set<int> chosen;
                chosen.reserve(K_meta * 2);
                while ((int)chosen.size() < K_meta) {
                    int c = sample_from_cdf(cdf, rng);
                    chosen.insert(c);
                }
                int j = 0;
                for (int c : chosen) {
                    sub_hnsw_tosearch[qi * K_meta + (j++)] = c;
                    if (j >= K_meta) break;
                }
            }
        }
    }


    int nthreads = omp_get_max_threads();
    std::vector<std::vector<std::unordered_set<int>>> local_sets(
        nthreads, std::vector<std::unordered_set<int>>(num_sub_hnsw));

    #pragma omp parallel for
    for (int i = 0; i < n * K_meta; i++) {
        int tid = omp_get_thread_num();
        int query_idx = i / K_meta;
        int sub_idx = sub_hnsw_tosearch[i];
        if (sub_idx >= 0 && sub_idx < num_sub_hnsw) {
            local_sets[tid][sub_idx].insert(query_idx);
        }
    }

    // merge
    std::unordered_map<int, std::unordered_set<int>> searchset;
    searchset.reserve(num_sub_hnsw);
    for (int sub_idx = 0; sub_idx < num_sub_hnsw; ++sub_idx) {
        for (int tid = 0; tid < nthreads; ++tid) {
            auto &ls = local_sets[tid][sub_idx];
            if (!ls.empty()) {
                auto &dst = searchset[sub_idx];
                dst.insert(ls.begin(), ls.end());
            }
        }
    }

    // sort by the number of queries that hit the sub (heavy-first)
    std::vector<std::pair<int,int>> searchset_vec;
    searchset_vec.reserve(searchset.size());
    for (auto &kv : searchset) {
        searchset_vec.emplace_back((int)kv.second.size(), kv.first); // (load, sub_idx)
    }
    std::sort(searchset_vec.begin(), searchset_vec.end(),
              [](auto const& a, auto const& b){ return a.first > b.first; });

    std::unordered_map<int, std::unordered_set<int>> reordered;
    reordered.reserve(searchset_vec.size());
    for (auto const &p : searchset_vec) {
        reordered.emplace(p.second, std::move(searchset[p.second]));
    }
    return reordered; 
}

std::unordered_map<int, std::unordered_set<int>> LocalHnsw::meta_search_pipelined(const int n,const float* query, int K_meta, float* distances, dhnsw_idx_t* labels, std::vector<int>& sub_hnsw_tosearch, int core_start, int cores_per_worker) {
    // std::cout << "meta_search_pipelined" << std::endl;
    
    // Acquire shared lock to prevent concurrent init() from modifying meta_hnsw
    // This protects against use-after-free when init() deletes and replaces meta_hnsw
    std::shared_lock<std::shared_mutex> lock(epoch_mutex_);
    
    omp_set_num_threads(cores_per_worker);
    meta_hnsw->search(n, query, K_meta, distances, labels);
    for(int i = 0; i < K_meta * n ; i++) {
        int label = labels[i];
        sub_hnsw_tosearch.push_back(label);
    }


    int nthreads = omp_get_max_threads();
    std::vector<std::vector<std::unordered_set<int>>> local_sets(nthreads,
        std::vector<std::unordered_set<int>>(num_sub_hnsw));

        #pragma omp parallel for
        for (int i = 0; i < n * K_meta; i++) {
            int tid = omp_get_thread_num();
            int query_idx = i / K_meta;
            int sub_idx = sub_hnsw_tosearch[i];
            local_sets[tid][sub_idx].insert(query_idx);
        }

        // merge to final result
        std::unordered_map<int, std::unordered_set<int>> searchset; 
        for (int sub_idx = 0; sub_idx < num_sub_hnsw; ++sub_idx) {
            for (int tid = 0; tid < nthreads; ++tid) {
                searchset[sub_idx].insert(
                    local_sets[tid][sub_idx].begin(),
                    local_sets[tid][sub_idx].end());
            }
            // if(searchset[sub_idx].size() > 0) {
            //     fetch_queue_.push(sub_idx);
            // }
    }
    // reorder searchset by num of queries
    std::vector<std::pair<int, int>> searchset_vec;
    for (auto const &kv : searchset) {
        searchset_vec.push_back({kv.second.size(), kv.first});
    }
    std::sort(searchset_vec.begin(), searchset_vec.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.first > b.first;
    });
    std::unordered_map<int, std::unordered_set<int>> reordered_searchset;
    for (auto const &p : searchset_vec) {
        reordered_searchset[p.second] = searchset[p.second];
    }
    searchset = reordered_searchset;
    return searchset;
}

// void LocalHnsw::fetch_loop() {
//     while (!stop_pipeline_) {
//         std::cout << "fetch_loop" << std::endl;
//         int sub_idx = fetch_queue_.pop();
//         if (sub_idx == -1){
//             if (stop_pipeline_) break;
//             std::this_thread::sleep_for(std::chrono::milliseconds(1));
//             continue;
//         }
//         if(cache_.check_and_evict()){
//             faiss::IndexHNSWFlat* idx  = fetch_sub_hnsw(sub_idx);
//             if (idx != nullptr) {  // Check for null before putting in cache
//                 cache_.put(sub_idx, idx);
//                 ready_queue_.push(sub_idx);
//             }
//         }
//         else{
//             std::cerr << "cache is full, evict failed" << std::endl;
//             fetch_queue_.push(sub_idx);
//             std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Brief pause
//         }
        
//     }
// }

// void LocalHnsw::search_loop() {
//     while (!stop_pipeline_) {
//         std::cout << "search_loop" << std::endl;
//         int sub_idx = ready_queue_.pop();
//         if (sub_idx == -1) {
//             // Check if we should really stop
//             if (stop_pipeline_) break;
//             std::this_thread::sleep_for(std::chrono::milliseconds(1));
//             continue;
//         }
//         faiss::IndexHNSWFlat* idx = cache_.get(sub_idx);
//         if (idx == nullptr) {
//             std::cerr << "sub_hnsw not found in cache" << std::endl;
//             continue;
//         }
//         auto it = pipelined_searchset_.find(sub_idx);
//         if (it == pipelined_searchset_.end()) {
//             // no queries for this shard (should not happen), just mark done:
//             if (--tasks_remaining_ == 0) {
//                 std::lock_guard<std::mutex> lg(done_mtx_);
//                 done_cv_.notify_one();
//             }
//             continue;
//         }

//         const std::unordered_set<int>& qset = it->second;
//         int m = (int)qset.size();
//         if (m == 0) {
//             int remaining = --tasks_remaining_;
//             if (remaining == 0) {
//                 std::lock_guard<std::mutex> lg(done_mtx_);
//                 done_cv_.notify_one();
//             }
//             continue;
//         }

//         // 3) Build the m × dim_ sub‐matrix of queries
//         std::vector<float> subQ(m * pipelined_dim_);
//         {
//             int i = 0;
//             for (int qi : qset) {
//                 // copy the qi‐th query into row i
//                 const float* src = pipelined_queries_ptr_ + (size_t)qi * pipelined_dim_;
//                 float* dst = subQ.data() + (size_t)i * pipelined_dim_;
//                 std::memcpy(dst, src, sizeof(float) * pipelined_dim_);
//                 ++i;
//             }
//         }

//         // 4) Run a batch‐search on idx_ptr
//         std::vector<float> tmpd(m * pipelined_K_sub_);
//         std::vector<dhnsw_idx_t> tmpl(m * pipelined_K_sub_);
//         idx->hnsw.efSearch = pipelined_efSearch_;
//         idx->search(m,
//                         subQ.data(),
//                         pipelined_K_sub_,
//                         tmpd.data(),
//                         tmpl.data());

//         // 5) Merge the results back into pipelined_distances_ptr_, pipelined_labels_ptr_, pipelined_tags_ptr_
//         {
//             std::lock_guard<std::mutex> lg(done_mtx_);
//             int i = 0;
//             for (int qi : qset) {
//                 float* outd = pipelined_distances_ptr_ + (size_t)qi * pipelined_K_sub_;
//                 dhnsw_idx_t* outl = pipelined_labels_ptr_ + (size_t)qi * pipelined_K_sub_;
//                 dhnsw_idx_t* outt = pipelined_tags_ptr_ + (size_t)qi * pipelined_K_sub_;
//                 for (int k = 0; k < pipelined_K_sub_; ++k) {
//                     float d = tmpd[(size_t)i * pipelined_K_sub_ + k];
//                     // insertion‐sort into outd/outl/outt
//                     if (d < outd[pipelined_K_sub_ - 1]) {
//                         int pos = pipelined_K_sub_ - 1;
//                         while (pos > 0 && outd[pos - 1] > d) {
//                             outd[pos] = outd[pos - 1];
//                             outl[pos] = outl[pos - 1];
//                             outt[pos] = outt[pos - 1];
//                             --pos;
//                         }
//                         outd[pos] = d;
//                         outl[pos] = tmpl[(size_t)i * pipelined_K_sub_ + k];
//                         outt[pos] = sub_idx;
//                     }
//                 }
//                 ++i;
//             }
//             // 6) Count this shard as completed
//             int remaining = --tasks_remaining_;
//             if (remaining == 0) {
//                 done_cv_.notify_one();
//             }
//         }
//     }
// }


// void LocalHnsw::start_search_pipeline(int num_search_threads) {
//     std::cout << "start_search_pipeline" << std::endl;
//     num_search_threads_ = num_search_threads;
//     stop_pipeline_ = false;
//     fetch_thread_ = std::thread(&LocalHnsw::fetch_loop, this);
//     for (int i = 0; i < num_search_threads; ++i) {
//         search_threads_.emplace_back(&LocalHnsw::search_loop, this);
//     }
// }

// void LocalHnsw::stop_search_pipeline() {
//     std::cout << "stop_search_pipeline" << std::endl;
//     stop_pipeline_ = true;
//     fetch_queue_.stop();
//     ready_queue_.stop();
//     if (fetch_thread_.joinable()) fetch_thread_.join();
//     for (auto &t : search_threads_) if (t.joinable()) t.join();
//     search_threads_.clear();
// }
// std::pair<double,double> LocalHnsw::sub_search_pipelined(
//         const int n,
//         const float* query,
//         int K_meta,
//         int K_sub,
//         float* distances,
//         dhnsw_idx_t* labels,
//         std::unordered_map<int, std::unordered_set<int>> searchset,
//         dhnsw_idx_t* sub_hnsw_tags,
//         int ef,
//         fetch_type flag,
//         int core_start,
//         int cores_per_worker)
// {
//     // std::cout << "sub_search_pipelined" << std::endl;


//     std::fill(distances, distances + (size_t)n * K_sub,
//               std::numeric_limits<float>::max());
//     std::fill(labels,    labels    + (size_t)n * K_sub, -1);
//     std::fill(sub_hnsw_tags, sub_hnsw_tags + (size_t)n * K_sub, -1);

//     double total_compute_time  = 0;
//     double total_network_latency = 0;

//     ThreadSafeQueue<int> batch_fetch_queue;
//     ThreadSafeQueue<int> batch_ready_queue;

//     int total_shards = (int)searchset.size();
//     std::atomic<int> batch_tasks_remaining{ total_shards };

//     std::mutex batch_done_mtx;
//     std::condition_variable batch_done_cv;

//     const float*  batch_queries_ptr   = query;
//     float*        batch_distances_ptr = distances;
//     dhnsw_idx_t*  batch_labels_ptr    = labels;
//     dhnsw_idx_t*  batch_tags_ptr      = sub_hnsw_tags;
//     int           batch_dim           = d;
//     int           batch_efSearch      = ef;
//     int           batch_K_sub_        = K_sub;

//     // Split searchset into “already in cache” vs. “must fetch”
//     std::vector<int> cached_indices;
//     std::vector<int> uncached_indices;
//     cached_indices.reserve(total_shards);
//     uncached_indices.reserve(total_shards);

//     // for (auto const &kv : searchset) {
//     //     int sub_idx = kv.first;
//     //     if (cache_.get(sub_idx) != nullptr) {
//     //         cached_indices.push_back(sub_idx);
//     //     } else {
//     //         uncached_indices.push_back(sub_idx);
//     //     }
//     // }
//      for (auto const &kv : searchset) {
//         int sub_idx = kv.first;
//         if (cache_.has_serialized_or_index(sub_idx)) {
//             cached_indices.push_back(sub_idx);
//         } else {
//             uncached_indices.push_back(sub_idx);
//         }
//     }

//     // in cache but not in searchset
//     for (auto const &kv : cache_.cache_map_) {
//         int sub_idx = kv.first;
//         if (searchset.count(sub_idx) == 0) {
//             cache_.used_indices_.insert(sub_idx);
//         }
//     }
//     //
//     // ─── FETCHER THREAD ────────────────────────────────────────────────────────────
//     //
//     auto fetch_worker = [&]() {
//         omp_set_num_threads(1);
//         while (true) {
//             int sub_idx = batch_fetch_queue.pop();
//             if (sub_idx < 0) {
//                 break;
//             }

//             while (!cache_.check_and_evict()) {
//                 // std::cout << "cache is full, evict failed; retrying..." << std::endl;
//                 //help update cache
//                 // check those has serialized data but do not have index
//                 cache_.idle_update();
//                 std::this_thread::sleep_for(std::chrono::milliseconds(1));
//             }
//             auto fetch_start = std::chrono::high_resolution_clock::now();
//             std::vector<uint8_t> serialized_data = fetch_sub_hnsw_pipelined(sub_idx);
//             // faiss::IndexHNSWFlat* idx = DistributedHnsw::deserialize_sub_hnsw(serialized_data);
//             auto fetch_end = std::chrono::high_resolution_clock::now();
//             auto fetch_duration = std::chrono::duration_cast<std::chrono::microseconds>(fetch_end - fetch_start).count();
//             // std::cout << "fetch_duration: " << fetch_duration << std::endl;
//             total_network_latency += fetch_duration;
//             // std::cout << "fetch done for sub_idx: " << sub_idx << std::endl;

//             if (serialized_data.size() > 0) {
//                 if (cache_.put(sub_idx, std::move(serialized_data))) {
//                     batch_ready_queue.push(sub_idx);
//                     // std::cout << "push ready_queue: " << sub_idx << std::endl;
//                 } else {
//                     // delete idx;
//                     std::cerr << "Failed to put index " << sub_idx << " in cache" << std::endl;
//                 }
//             }
//         }

//         batch_ready_queue.push(-1); //signal to search_thread to stop
//         batch_ready_queue.push(-1); //signal to fetch_thread who start search to stop
//         // std::cout << "fetch_worker done, start join search_thread" << std::endl;
//         cpu_set_t new_set;
//         CPU_ZERO(&new_set);
//         for (int i = 1; i < cores_per_worker; i++) {
//             CPU_SET(core_start + i*2, &new_set);
//         }
//         pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &new_set);
//          while (true) {
//             int sub_idx = batch_ready_queue.pop();
//             if (sub_idx < 0) {
//                 break;
//             }

//             {
//             std::lock_guard<std::mutex> lg(cache_mutex_);
//             cache_.in_use_indices_.insert(sub_idx);
//             }

//             std::pair<std::vector<uint8_t>, faiss::IndexHNSWFlat*> serialized_and_index = cache_.get_serialized_and_index(sub_idx);
//             std::vector<uint8_t> serialized_data = serialized_and_index.first;
//             faiss::IndexHNSWFlat* idx = serialized_and_index.second;
//             if (serialized_data.size() == 0) {
//                 std::cerr << "sub_hnsw not found in cache for " << sub_idx << std::endl;
//                 // Even if it’s missing, we still COUNT this shard as “done”:
//                 int rem = --batch_tasks_remaining;
//                 if (rem == 0) {
//                     std::lock_guard<std::mutex> lg(batch_done_mtx);
//                     batch_done_cv.notify_one();
//                 }
//                 continue;
//             }

//             // Look up which queries go to this shard
//             auto it = searchset.find(sub_idx);
//             if (it == searchset.end() || it->second.empty()) {
//                 // No queries for sub_idx: still count as “finished”
//                 int rem = --batch_tasks_remaining;
//                 if (rem == 0) {
//                     std::lock_guard<std::mutex> lg(batch_done_mtx);
//                     batch_done_cv.notify_one();
//                 }
//                 continue;
//             }

//             const std::unordered_set<int>& qset = it->second;
//             int m = (int)qset.size();

//             // Build a local “subQ” of size m × dim
//             std::vector<float> subQ((size_t)m * batch_dim);
//             {
//                 int i = 0;
//                 for (int qi : qset) {
//                     const float* src = batch_queries_ptr + (size_t)qi * batch_dim;
//                     float* dst       = subQ.data() + (size_t)i * batch_dim;
//                     std::memcpy(dst, src, sizeof(float) * batch_dim);
//                     ++i;
//                 }
//             }

//             // Run the actual HNSW search on “m” queries
//             std::vector<float> tmpd((size_t)m * batch_K_sub_);
//             std::vector<dhnsw_idx_t> tmpl((size_t)m * batch_K_sub_);
//             if (idx == nullptr) {
//                 idx = DistributedHnsw::deserialize_sub_hnsw(serialized_data);
//                 cache_.update(sub_idx, idx);
//             }
//             idx->hnsw.efSearch = batch_efSearch;
//             idx->search(m, subQ.data(), batch_K_sub_, tmpd.data(), tmpl.data());
//             {
//             std::lock_guard<std::mutex> lg(cache_mutex_);
//             cache_.in_use_indices_.erase(sub_idx);
//             }

//             // std::cout << "search done for sub_idx: " << sub_idx << std::endl;
//             // Merge results back into “distances/labels/sub_hnsw_tags”
//             {
//                 std::lock_guard<std::mutex> lg(batch_done_mtx);
//                 int i = 0;
//                 for (int qi : qset) {
//                     float*    outd = batch_distances_ptr + (size_t)qi * batch_K_sub_;
//                     dhnsw_idx_t* outl = batch_labels_ptr + (size_t)qi * batch_K_sub_;
//                     dhnsw_idx_t* outt = batch_tags_ptr + (size_t)qi * batch_K_sub_;

//                     for (int k = 0; k < batch_K_sub_; ++k) {
//                         float d = tmpd[(size_t)i * batch_K_sub_ + k];
//                         if (d < outd[batch_K_sub_ - 1]) {
//                             int pos = batch_K_sub_ - 1;
//                             while (pos > 0 && outd[pos - 1] > d) {
//                                 outd[pos]  = outd[pos - 1];
//                                 outl[pos]  = outl[pos - 1];
//                                 outt[pos]  = outt[pos - 1];
//                                 --pos;
//                             }
//                             outd[pos] = d;
//                             outl[pos] = tmpl[(size_t)i * batch_K_sub_ + k];
//                             outt[pos] = sub_idx;
//                         }
//                     }
//                     ++i;
//                 }

//                 int rem = --batch_tasks_remaining;
//                 if (rem == 0) {
//                     batch_done_cv.notify_one();
//                 }
//             }
//             // std::cout << "search_worker done for sub_idx: " << sub_idx << std::endl;
//         }

//     };

//     //
//     // ─── SEARCHER THREAD ────────────────────────────────────────────────────────────
//     //
//     auto search_worker = [&]() {
//         omp_set_num_threads(cores_per_worker - 1);
//         while (true) {
//             int sub_idx = batch_ready_queue.pop();
//             if (sub_idx < 0) {
//                 break;
//             }

//              {
//             std::lock_guard<std::mutex> lg(cache_mutex_);
//             cache_.in_use_indices_.insert(sub_idx);
//             }

//             std::pair<std::vector<uint8_t>, faiss::IndexHNSWFlat*> serialized_and_index = cache_.get_serialized_and_index(sub_idx);
//             std::vector<uint8_t> serialized_data = serialized_and_index.first;
//             faiss::IndexHNSWFlat* idx = serialized_and_index.second;
//             if (serialized_data.empty() && idx == nullptr) {
//                 std::cerr << "sub_hnsw not found in cache for " << sub_idx << std::endl;
//                 // Even if it’s missing, we still COUNT this shard as “done”:
//                 int rem = --batch_tasks_remaining;
//                 if (rem == 0) {
//                     std::lock_guard<std::mutex> lg(batch_done_mtx);
//                     batch_done_cv.notify_one();
//                 }
//                 continue;
//             }
//             auto search_start = std::chrono::high_resolution_clock::now();
//             if (idx == nullptr) {
//                 idx = DistributedHnsw::deserialize_sub_hnsw(serialized_data);
//                 cache_.update(sub_idx, idx);
//             }
//             cache_.used_indices_.insert(sub_idx); //check later
//             // Look up which queries go to this shard         
//             auto it = searchset.find(sub_idx);

//             const std::unordered_set<int>& qset = it->second;
//             int m = (int)qset.size();

//             // Build a local “subQ” of size m × dim
//             std::vector<float> subQ((size_t)m * batch_dim);
//             {
//                 int i = 0;
//                 for (int qi : qset) {
//                     const float* src = batch_queries_ptr + (size_t)qi * batch_dim;
//                     float* dst       = subQ.data() + (size_t)i * batch_dim;
//                     std::memcpy(dst, src, sizeof(float) * batch_dim);
//                     ++i;
//                 }
//             }

//             // Run the actual HNSW search on “m” queries
//             std::vector<float> tmpd((size_t)m * batch_K_sub_);
//             std::vector<dhnsw_idx_t> tmpl((size_t)m * batch_K_sub_);

//             idx->hnsw.efSearch = batch_efSearch;
//             idx->search(m, subQ.data(), batch_K_sub_, tmpd.data(), tmpl.data());
//             {
//             std::lock_guard<std::mutex> lg(cache_mutex_);
//             cache_.in_use_indices_.erase(sub_idx);
//             }

//             // std::cout << "search done for sub_idx: " << sub_idx << std::endl;
//             // Merge results back into “distances/labels/sub_hnsw_tags”
//             {
//                 std::lock_guard<std::mutex> lg(batch_done_mtx);
//                 int i = 0;
//                 for (int qi : qset) {
//                     float*    outd = batch_distances_ptr + (size_t)qi * batch_K_sub_;
//                     dhnsw_idx_t* outl = batch_labels_ptr + (size_t)qi * batch_K_sub_;
//                     dhnsw_idx_t* outt = batch_tags_ptr + (size_t)qi * batch_K_sub_;

//                     for (int k = 0; k < batch_K_sub_; ++k) {
//                         float d = tmpd[(size_t)i * batch_K_sub_ + k];
//                         if (d < outd[batch_K_sub_ - 1]) {
//                             int pos = batch_K_sub_ - 1;
//                             while (pos > 0 && outd[pos - 1] > d) {
//                                 outd[pos]  = outd[pos - 1];
//                                 outl[pos]  = outl[pos - 1];
//                                 outt[pos]  = outt[pos - 1];
//                                 --pos;
//                             }
//                             outd[pos] = d;
//                             outl[pos] = tmpl[(size_t)i * batch_K_sub_ + k];
//                             outt[pos] = sub_idx;
//                         }
//                     }
//                     ++i;
//                 }

//                 int rem = --batch_tasks_remaining;
//                 if (rem == 0) {
//                     batch_done_cv.notify_one();
//                 }
//             }
//             auto search_end = std::chrono::high_resolution_clock::now();
//             auto search_duration = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start).count();
//             // std::cout << "search_duration: " << search_duration << std::endl;
//             total_compute_time += search_duration;
//             // std::cout << "search_worker done for sub_idx: " << sub_idx << std::endl;
//         }
//     };

//     // ─── Actually start the two threads ───────────────────────────────────────────
//     std::thread fetch_thread(fetch_worker);
//     std::thread search_thread(search_worker);


//     cpu_set_t fetch_set, search_set;
//     CPU_ZERO(&fetch_set);
//     CPU_ZERO(&search_set);

//     // Pin fetch_thread to the **first** core only:
//     CPU_SET(core_start + 0, &fetch_set);
//     pthread_setaffinity_np(fetch_thread.native_handle(),
//                            sizeof(cpu_set_t), &fetch_set);

//     // Pin search_thread to **all the remaining** 8 cores:
//     for (int i = 1; i < cores_per_worker; i++) {
//         CPU_SET(core_start + i*2, &search_set);
//     }
//     pthread_setaffinity_np(search_thread.native_handle(),
//                            sizeof(cpu_set_t), &search_set);
//     for (int sub_idx : cached_indices) {
//         {
//             std::lock_guard<std::mutex> lg(cache_mutex_);
//             cache_.in_use_indices_.insert(sub_idx);
//         }
//         batch_ready_queue.push(sub_idx);
//         // std::cout << "push cached_indices: " << sub_idx << std::endl;
//     }

//     for (int sub_idx : uncached_indices) {
//         batch_fetch_queue.push(sub_idx);
//         // std::cout << "push uncached_indices: " << sub_idx << std::endl;
//     }

//     batch_fetch_queue.push(-1);

   
//     {
//         std::unique_lock<std::mutex> lock(batch_done_mtx);
//         bool completed = batch_done_cv.wait_for(
//             lock,
//             std::chrono::seconds(10),
//             [&]() { return batch_tasks_remaining.load() == 0; }
//         );

//         if (!completed) {
//             std::cerr << "Batch timeout - forcing completion" << std::endl;
//             batch_fetch_queue.stop();
//             batch_ready_queue.stop();
//         }
//     }

//     // By now, tasks_remaining == 0 exactly once.  Both threads must have seen their single “−1”:
//     if (fetch_thread.joinable())  fetch_thread.join();
//     if (search_thread.joinable()) search_thread.join();
//     cache_.used_indices_.clear();
//     cache_.in_use_indices_.clear();
//     // std::cout << "sub_search_pipelined done" << std::endl;
//     return { total_compute_time, total_network_latency };
// }
std::tuple<double,double,double> LocalHnsw::sub_search_pipelined(
            const int n, const float* query, int K_meta, int K_sub, 
            float* distances, dhnsw_idx_t* labels, 
            std::unordered_map<int, std::unordered_set<int>> searchset,
            dhnsw_idx_t* sub_hnsw_tags,
            int ef, fetch_type flag, int core_start, int cores_per_worker){
    std::fill(distances, distances + (size_t)n * K_sub,
              std::numeric_limits<float>::max());
    std::fill(labels,    labels    + (size_t)n * K_sub, -1);
    std::fill(sub_hnsw_tags, sub_hnsw_tags + (size_t)n * K_sub, -1);

    double total_compute_time  = 0;
    double total_network_latency = 0;
    double total_deserialize_time = 0;
    ThreadSafeQueue<int> batch_fetch_queue; //first
    ThreadSafeQueue<int> batch_deserialize_queue; //second

    int total_shards = (int)searchset.size();
    std::atomic<int> batch_tasks_remaining{ total_shards };

    std::mutex batch_done_mtx;
    std::condition_variable batch_done_cv;

    const float*  batch_queries_ptr   = query;
    float*        batch_distances_ptr = distances;
    dhnsw_idx_t*  batch_labels_ptr    = labels;
    dhnsw_idx_t*  batch_tags_ptr      = sub_hnsw_tags;
    int           batch_dim           = d;
    int           batch_efSearch      = ef;
    int           batch_K_sub_        = K_sub;

    // Split searchset into “already in cache” vs. “must fetch”
    std::vector<int> cached_indices;
    std::vector<int> uncached_indices;
    cached_indices.reserve(total_shards);
    uncached_indices.reserve(total_shards);
    // for (auto const &kv : searchset) {
    //     int sub_idx = kv.first;
    //     if (cache_.get(sub_idx) != nullptr) {
    //         cached_indices.push_back(sub_idx);
    //     } else {
    //         uncached_indices.push_back(sub_idx);
    //     }
    // }
     for (auto const &kv : searchset) {
        int sub_idx = kv.first;
        if (cache_.has_serialized_or_index(sub_idx)) {
            cached_indices.push_back(sub_idx);
        } else {
            uncached_indices.push_back(sub_idx);
        }
    }

    // in cache but not in searchset - mark as used (with proper locking)
    for (auto const &kv : cache_.cache_map_) {
        int sub_idx = kv.first;
        if (searchset.count(sub_idx) == 0) {
            cache_.mark_used(sub_idx);  // Thread-safe version (fixes race condition)
        }
    }
    //
    // ─── FETCHER THREAD ────────────────────────────────────────────────────────────
    //
    auto fetch_worker = [&]() {
        omp_set_num_threads(1);
        // std::cout << "fetch_worker start" << std::endl;
        while (true) {
            int sub_idx = batch_fetch_queue.pop();
            if (sub_idx < 0) {
                batch_deserialize_queue.push(-1); // check later
                break;
            }

            while (!cache_.evict_one()) {
                // std::cout << "cache is full, evict failed; retrying..." << std::endl;
                //help update cache
                // check those has serialized data but do not have index
                // cache_.idle_update();
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            auto fetch_start = std::chrono::high_resolution_clock::now();
            std::vector<uint8_t> serialized_data = fetch_sub_hnsw_pipelined(sub_idx);
            // faiss::IndexHNSWFlat* idx = DistributedHnsw::deserialize_sub_hnsw(serialized_data);
            auto fetch_end = std::chrono::high_resolution_clock::now();
            total_network_latency += std::chrono::duration_cast<std::chrono::microseconds>(fetch_end - fetch_start).count();
            // std::cout << "fetch done for sub_idx: " << sub_idx << std::endl;
            if (!serialized_data.empty()) {
                bool ok = cache_.put(sub_idx, std::move(serialized_data));
                cache_.mark_in_use(sub_idx);
                if (!ok) {
                    std::cerr << "LRUCache::put failed for sub_idx=" << sub_idx << std::endl;
                } else {
                    batch_deserialize_queue.push(sub_idx);
                }
            } else {
                std::cerr << "fetch_sub_hnsw_pipelined failed for sub_idx=" << sub_idx << std::endl;
                int rem = --batch_tasks_remaining;
                if (rem == 0) {
                    std::lock_guard<std::mutex> lg(batch_done_mtx);
                    batch_done_cv.notify_one();
                }
            }
        }

        
       
    };
    //
    // ─── SEARCHER THREAD ────────────────────────────────────────────────────────────
    //
    auto search_worker = [&]() {
        omp_set_num_threads(cores_per_worker-1);
        // std::cout << "search_worker start" << std::endl;
        while (true) {
            int sub_idx = batch_deserialize_queue.pop();
            if (sub_idx < 0) {
                //done
                break;
            }

            std::pair<std::shared_ptr<std::vector<uint8_t>>, std::shared_ptr<faiss::IndexHNSWFlat>> entry
                = cache_.get_serialized_and_index(sub_idx);
            std::shared_ptr<std::vector<uint8_t>> serialized_data = entry.first;
            std::shared_ptr<faiss::IndexHNSWFlat> idx_ptr = entry.second;

            if (serialized_data->empty()) {
                // Shouldn't happen unless there's a race—but count it as done regardless
                std::cerr << "serialized_data is empty for sub_idx=" << sub_idx << std::endl;
                int rem = --batch_tasks_remaining;
                if (rem == 0) {
                    std::lock_guard<std::mutex> lg(batch_done_mtx);
                    batch_done_cv.notify_one();
                }
                continue;
            }

            if (!idx_ptr) {
                auto deser_start = std::chrono::high_resolution_clock::now();
                faiss::IndexHNSWFlat* raw_idx = deserialize_sub_hnsw_pipelined(*serialized_data);
                auto deser_end = std::chrono::high_resolution_clock::now();
                total_deserialize_time += std::chrono::duration_cast<std::chrono::microseconds>(deser_end - deser_start).count();

                // Store into cache as shared_ptr
                idx_ptr = std::shared_ptr<faiss::IndexHNSWFlat>(raw_idx);
                cache_.update(sub_idx, idx_ptr);
            }
            cache_.mark_in_use(sub_idx);
            cache_.mark_used(sub_idx);  // Thread-safe version (fixes race condition)
            auto search_start = std::chrono::high_resolution_clock::now();
            
           
            // Look up which queries go to this shard         
            auto it = searchset.find(sub_idx);

            const std::unordered_set<int>& qset = it->second;
            int m = (int)qset.size();
            if (m == 0) {
                std::cerr << "qset is empty for sub_idx=" << sub_idx << std::endl;
                int rem = --batch_tasks_remaining;
                if (rem == 0) {
                    std::lock_guard<std::mutex> lg(batch_done_mtx);
                    batch_done_cv.notify_one();
                }
                continue;
            }
            // Build a local “subQ” of size m × dim
            std::vector<float> subQ((size_t)m * batch_dim);
            {
                int i = 0;
                for (int qi : qset) {
                    const float* src = batch_queries_ptr + (size_t)qi * batch_dim;
                    float* dst       = subQ.data() + (size_t)i * batch_dim;
                    std::memcpy(dst, src, sizeof(float) * batch_dim);
                    ++i;
                }
            }

            // Run the actual HNSW search on “m” queries
            std::vector<float> tmpd((size_t)m * batch_K_sub_);
            std::vector<dhnsw_idx_t> tmpl((size_t)m * batch_K_sub_);

            idx_ptr->hnsw.efSearch = batch_efSearch;
            idx_ptr->search(m, subQ.data(), batch_K_sub_, tmpd.data(), tmpl.data());
            cache_.unmark_in_use(sub_idx);

            // std::cout << "search done for sub_idx: " << sub_idx << std::endl;
            // Merge results back into “distances/labels/sub_hnsw_tags”
            {
                std::lock_guard<std::mutex> lg(batch_done_mtx);
                int i = 0;
                for (int qi : qset) {
                    float*    outd = batch_distances_ptr + (size_t)qi * batch_K_sub_;
                    dhnsw_idx_t* outl = batch_labels_ptr + (size_t)qi * batch_K_sub_;
                    dhnsw_idx_t* outt = batch_tags_ptr + (size_t)qi * batch_K_sub_;

                    for (int k = 0; k < batch_K_sub_; ++k) {
                        float d = tmpd[(size_t)i * batch_K_sub_ + k];
                        if (d < outd[batch_K_sub_ - 1]) {
                            int pos = batch_K_sub_ - 1;
                            while (pos > 0 && outd[pos - 1] > d) {
                                outd[pos]  = outd[pos - 1];
                                outl[pos]  = outl[pos - 1];
                                outt[pos]  = outt[pos - 1];
                                --pos;
                            }
                            outd[pos] = d;
                            outl[pos] = tmpl[(size_t)i * batch_K_sub_ + k];
                            outt[pos] = sub_idx;
                        }
                    }
                    ++i;
                }

                int rem = --batch_tasks_remaining;
                if (rem == 0) {
                    batch_done_cv.notify_one();
                }
            }
            auto search_end = std::chrono::high_resolution_clock::now();
            total_compute_time += std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start).count();
            // std::cout << "search_worker done for sub_idx: " << sub_idx << std::endl;
        }
    };

    // ─── Actually start the two threads ───────────────────────────────────────────
    std::thread fetch_thread(fetch_worker);
    std::thread search_thread(search_worker);

    cpu_set_t fetch_set, search_set;
    CPU_ZERO(&fetch_set);
    CPU_ZERO(&search_set);

    // Pin fetch_thread to the first core only:
    CPU_SET(core_start + 0, &fetch_set);
    pthread_setaffinity_np(fetch_thread.native_handle(),
                           sizeof(cpu_set_t), &fetch_set);

    // Pin search_thread to **all the remaining** 8 cores:
    for (int i = 1; i < cores_per_worker; i++) {
        CPU_SET(core_start + i*2, &search_set);
    }
    pthread_setaffinity_np(search_thread.native_handle(),
                           sizeof(cpu_set_t), &search_set);
    for (int sub_idx : cached_indices) {
        cache_.mark_in_use(sub_idx);  // Thread-safe version using cache's internal mutex
        batch_deserialize_queue.push(sub_idx);
        // std::cout << "push cached_indices: " << sub_idx << std::endl;
    }

    for (int sub_idx : uncached_indices) {
        batch_fetch_queue.push(sub_idx);
        // std::cout << "push uncached_indices: " << sub_idx << std::endl;
    }

    batch_fetch_queue.push(-1);

   
    {
        std::unique_lock<std::mutex> lock(batch_done_mtx);
        bool completed = batch_done_cv.wait_for(
            lock,
            std::chrono::seconds(10),
            [&]() { return batch_tasks_remaining.load() == 0; }
        );

        if (!completed) {
            std::cerr << "Batch timeout - forcing completion" << std::endl;
            batch_fetch_queue.stop();
            batch_deserialize_queue.stop();
        }
    }

    // By now, tasks_remaining == 0 exactly once.  Both threads must have seen their single "−1":
    if (fetch_thread.joinable())  fetch_thread.join();
    if (search_thread.joinable()) search_thread.join();
    cache_.clear_usage_tracking();  // Thread-safe version (fixes race condition)
    // std::cout << "sub_search_pipelined done" << std::endl;
    return std::make_tuple(total_compute_time, total_network_latency, total_deserialize_time);           
}

std::tuple<double,double,double> LocalHnsw::sub_search_pipelined_3_stage(
        const int n,
        const float* query,
        int K_meta,
        int K_sub,
        float* distances,
        dhnsw_idx_t* labels,
        std::unordered_map<int, std::unordered_set<int>> searchset,
        dhnsw_idx_t* sub_hnsw_tags,
        int ef,
        fetch_type flag,
        int core_start,
        int cores_per_worker)
{
    // std::cout << "sub_search_pipelined" << std::endl;

    auto pipeline_start_time = std::chrono::high_resolution_clock::now();
    std::fill(distances, distances + (size_t)n * K_sub,
              std::numeric_limits<float>::max());
    std::fill(labels,    labels    + (size_t)n * K_sub, -1);
    std::fill(sub_hnsw_tags, sub_hnsw_tags + (size_t)n * K_sub, -1);

    double total_compute_time  = 0;
    double total_network_latency = 0;
    double total_deserialize_time = 0;
    ThreadSafeQueue<int> batch_fetch_queue; //first
    ThreadSafeQueue<int> batch_ready_queue; //third
    ThreadSafeQueue<int> batch_deserialize_queue; //second

    int total_shards = (int)searchset.size();
    std::atomic<int> batch_tasks_remaining{ total_shards };

    std::mutex batch_done_mtx;
    std::condition_variable batch_done_cv;

    const float*  batch_queries_ptr   = query;
    float*        batch_distances_ptr = distances;
    dhnsw_idx_t*  batch_labels_ptr    = labels;
    dhnsw_idx_t*  batch_tags_ptr      = sub_hnsw_tags;
    int           batch_dim           = d;
    int           batch_efSearch      = ef;
    int           batch_K_sub_        = K_sub;

    // Split searchset into “already in cache” vs. “must fetch”
    std::vector<int> cached_indices;
    std::vector<int> uncached_indices;
    std::vector<int> cached_not_deserialize_indices;
    cached_indices.reserve(total_shards);
    uncached_indices.reserve(total_shards);
    cached_not_deserialize_indices.reserve(total_shards);
    // for (auto const &kv : searchset) {
    //     int sub_idx = kv.first;
    //     if (cache_.get(sub_idx) != nullptr) {
    //         cached_indices.push_back(sub_idx);
    //     } else {
    //         uncached_indices.push_back(sub_idx);
    //     }
    // }
     for (auto const &kv : searchset) {
        int sub_idx = kv.first;
        if (cache_.has_index(sub_idx)) {
            cached_indices.push_back(sub_idx);
        } else if (cache_.has_serialized_not_index(sub_idx)) {
            cached_not_deserialize_indices.push_back(sub_idx);
        } else {
            uncached_indices.push_back(sub_idx);
        }
    }

    // in cache but not in searchset - mark as used (with proper locking)
    for (auto const &kv : cache_.cache_map_) {
        int sub_idx = kv.first;
        if (searchset.count(sub_idx) == 0) {
            cache_.mark_used(sub_idx);  // Thread-safe version (fixes race condition)
        }
    }
    //
    // ─── FETCHER THREAD ────────────────────────────────────────────────────────────
    //
    auto fetch_worker = [&]() {
        omp_set_num_threads(1);
        // std::cout << "fetch_worker start" << std::endl;
        while (true) {
            int sub_idx = batch_fetch_queue.pop();
            if (sub_idx < 0) {
                batch_deserialize_queue.push(-1); // check later
                break;
            }

            int items_to_fetch = cache_.capacity_;
            for (int i = 0; i < items_to_fetch; ++i) {
                while (!cache_.evict_one()) { 
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            }

            std::pair<bool, double> fetch_result;
            int available = cache_.capacity_ - cache_.cache_map_.size();

            if(cache_.cache_map_.size() <= 1 || available == 1){
                auto fetch_trace_start = std::chrono::high_resolution_clock::now();
                fetch_result = fetch_sub_hnsw_and_put_pipelined(sub_idx, batch_deserialize_queue);
                auto fetch_trace_end = std::chrono::high_resolution_clock::now();
                // g_tracer.add_event({
                //     "Fetch", 
                //     {sub_idx}, 
                //     std::chrono::duration_cast<std::chrono::microseconds>(fetch_trace_start - pipeline_start_time).count(),
                //     std::chrono::duration_cast<std::chrono::microseconds>(fetch_trace_end - pipeline_start_time).count()
                // });
            }
            else{
                //pop capacity - size and doorbell fetch
                bool stop = false;
                std::vector<int> sub_indices;
                sub_indices.push_back(sub_idx);
                for (int i = 1; i < available; i++) {
                    int tmp = batch_fetch_queue.pop();
                    if (tmp < 0) {
                        stop = true;
                        break;
                    }
                    sub_indices.push_back(tmp);
                }
                auto fetch_trace_start = std::chrono::high_resolution_clock::now();
                fetch_result = fetch_sub_hnsw_batch_with_doorbell_and_put_pipelined(sub_indices, batch_deserialize_queue, stop);
                auto fetch_trace_end = std::chrono::high_resolution_clock::now();
                // g_tracer.add_event({
                //     "Fetch", 
                //     sub_indices, 
                //     std::chrono::duration_cast<std::chrono::microseconds>(fetch_trace_start - pipeline_start_time).count(),
                //     std::chrono::duration_cast<std::chrono::microseconds>(fetch_trace_end - pipeline_start_time).count()
                // });
            }
            // faiss::IndexHNSWFlat* idx = DistributedHnsw::deserialize_sub_hnsw(serialized_data);
          
            // std::cout << "fetch_duration: " << fetch_duration << std::endl;
            // std::cout << "fetch done for sub_idx: " << sub_idx << std::endl;

            if (!fetch_result.first) {
                std::cerr << "Failed to put index " << sub_idx << " in cache" << std::endl;
            }
            total_network_latency += fetch_result.second;
            // std::cout << "fetch for subhnsw:" << sub_idx<< "is" << fetch_result.second<<std::endl;
                    // std::pair<bool, double> fetch_result = fetch_sub_hnsw_and_put_pipelined(sub_idx, batch_deserialize_queue);
                    // if (fetch_result.first) {
                    //     total_network_latency += fetch_result.second;
                    //     std::cout << "fetch for subhnsw:" << sub_idx<< "is" << fetch_result.second<<std::endl;
                    // }else{
                    //     std::cerr << "Failed to fetch index " << sub_idx << std::endl;
                    // }
                    // std::cout << "fetch done for sub_idx: " << sub_idx << std::endl;

                }

                
            
            };




    //
    // ─── DESERIALIZE THREAD ────────────────────────────────────────────────────────────
    //
    auto deserialize_worker = [&]() {
        omp_set_num_threads(1);
        // std::cout << "deserialize_worker start" << std::endl;
        while (true) {
            int sub_idx = batch_deserialize_queue.pop();
            if (sub_idx < 0) {
                batch_ready_queue.push(-1); // check later
                break;
            }

            std::pair<std::shared_ptr<std::vector<uint8_t>>, std::shared_ptr<faiss::IndexHNSWFlat>> serialized_and_index = cache_.get_serialized_and_index(sub_idx);
            std::shared_ptr<std::vector<uint8_t>> serialized_data = serialized_and_index.first;
            std::shared_ptr<faiss::IndexHNSWFlat> idx = serialized_and_index.second;
            if (serialized_data->size() == 0) {
                std::cerr << "sub_hnsw not found in cache for " << sub_idx << std::endl;
                continue;
            }   
            if (idx == nullptr) {
                auto deserialize_start = std::chrono::high_resolution_clock::now();
                faiss::IndexHNSWFlat* idx_tmp = deserialize_sub_hnsw_pipelined(*serialized_data);
                auto deserialize_end = std::chrono::high_resolution_clock::now();
                // g_tracer.add_event({
                //     "deserailze", 
                //     {sub_idx}, 
                //     std::chrono::duration_cast<std::chrono::microseconds>(deserialize_start - pipeline_start_time).count(),
                //     std::chrono::duration_cast<std::chrono::microseconds>(deserialize_end - pipeline_start_time).count()
                // });          
                auto deserialize_duration = std::chrono::duration_cast<std::chrono::microseconds>(deserialize_end - deserialize_start).count();
                // std::cout << "deserialize_duration: " << deserialize_duration << std::endl;
                total_deserialize_time += deserialize_duration;
                cache_.update(sub_idx, std::shared_ptr<faiss::IndexHNSWFlat>(idx_tmp));
                batch_ready_queue.push(sub_idx);
                // std::cout << "deserialize done for sub_idx: " << sub_idx << std::endl;
            }else{
                //already have index
                batch_ready_queue.push(sub_idx);
            }        
        }
    };
    //
    // ─── SEARCHER THREAD ────────────────────────────────────────────────────────────
    //
    auto search_worker = [&]() {
        omp_set_num_threads(7);
        // std::cout << "search_worker start" << std::endl;
        while (true) {
            int sub_idx = batch_ready_queue.pop();
            if (sub_idx < 0) {
                //done
                break;
            }

            // Use atomic get+mark_in_use to prevent use-after-free race condition
            std::shared_ptr<faiss::IndexHNSWFlat> idx = cache_.get_index_ptr_and_mark_in_use(sub_idx);
            if (idx == nullptr) {
                std::cerr << "sub_hnsw not found in cache for " << sub_idx << std::endl;
                // Even if it's missing, we still COUNT this shard as "done":
                int rem = --batch_tasks_remaining;
                if (rem == 0) {
                    std::lock_guard<std::mutex> lg(batch_done_mtx);
                    batch_done_cv.notify_one();
                }
                continue;
            }
            auto search_start = std::chrono::high_resolution_clock::now();
            
            cache_.mark_used(sub_idx);  // Thread-safe version (fixes race condition)
            // Look up which queries go to this shard         
            auto it = searchset.find(sub_idx);

            const std::unordered_set<int>& qset = it->second;
            int m = (int)qset.size();

            // Build a local "subQ" of size m × dim
            std::vector<float> subQ((size_t)m * batch_dim);
            {
                int i = 0;
                for (int qi : qset) {
                    const float* src = batch_queries_ptr + (size_t)qi * batch_dim;
                    float* dst       = subQ.data() + (size_t)i * batch_dim;
                    std::memcpy(dst, src, sizeof(float) * batch_dim);
                    ++i;
                }
            }

            // Run the actual HNSW search on “m” queries
            std::vector<float> tmpd((size_t)m * batch_K_sub_);
            std::vector<dhnsw_idx_t> tmpl((size_t)m * batch_K_sub_);

            idx->hnsw.efSearch = batch_efSearch;
            idx->search(m, subQ.data(), batch_K_sub_, tmpd.data(), tmpl.data());
            cache_.unmark_in_use(sub_idx);

            // std::cout << "search done for sub_idx: " << sub_idx << std::endl;
            // Merge results back into “distances/labels/sub_hnsw_tags”
            {
                std::lock_guard<std::mutex> lg(batch_done_mtx);
                int i = 0;
                for (int qi : qset) {
                    float*    outd = batch_distances_ptr + (size_t)qi * batch_K_sub_;
                    dhnsw_idx_t* outl = batch_labels_ptr + (size_t)qi * batch_K_sub_;
                    dhnsw_idx_t* outt = batch_tags_ptr + (size_t)qi * batch_K_sub_;

                    for (int k = 0; k < batch_K_sub_; ++k) {
                        float d = tmpd[(size_t)i * batch_K_sub_ + k];
                        if (d < outd[batch_K_sub_ - 1]) {
                            int pos = batch_K_sub_ - 1;
                            while (pos > 0 && outd[pos - 1] > d) {
                                outd[pos]  = outd[pos - 1];
                                outl[pos]  = outl[pos - 1];
                                outt[pos]  = outt[pos - 1];
                                --pos;
                            }
                            outd[pos] = d;
                            outl[pos] = tmpl[(size_t)i * batch_K_sub_ + k];
                            outt[pos] = sub_idx;
                        }
                    }
                    ++i;
                }

                int rem = --batch_tasks_remaining;
                if (rem == 0) {
                    batch_done_cv.notify_one();
                }
            }
            auto search_end = std::chrono::high_resolution_clock::now();
            // g_tracer.add_event({
            //         "search", 
            //         {sub_idx}, 
            //         std::chrono::duration_cast<std::chrono::microseconds>(search_start - pipeline_start_time).count(),
            //         std::chrono::duration_cast<std::chrono::microseconds>(search_end - pipeline_start_time).count()
            //     });     
            auto search_duration = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start).count();
            // std::cout << "search_duration: " << search_duration << std::endl;
            total_compute_time += search_duration;
            // std::cout << "search_worker done for sub_idx: " << sub_idx << std::endl;
        }
    };

    // ─── Actually start the two threads ───────────────────────────────────────────
    std::thread fetch_thread(fetch_worker);
    std::thread search_thread(search_worker);
    std::thread deserialize_thread(deserialize_worker);

    cpu_set_t fetch_set, search_set, deserialize_set;
    CPU_ZERO(&fetch_set);
    CPU_ZERO(&search_set);
    CPU_ZERO(&deserialize_set);

    // Pin fetch_thread to the first core only:
    CPU_SET(core_start + 0, &fetch_set);
    pthread_setaffinity_np(fetch_thread.native_handle(),
                           sizeof(cpu_set_t), &fetch_set);
    // Pin deserialize_thread to the first hyperthread only:
    for (int i = 1; i < 2; i++) {
        CPU_SET(core_start + i*2 , &deserialize_set);
    } 
    pthread_setaffinity_np(deserialize_thread.native_handle(),
                           sizeof(cpu_set_t), &deserialize_set);

    // Pin search_thread to **all the remaining** 8 cores:
    for (int i = 2; i < cores_per_worker; i++) {
        CPU_SET(core_start + i*2, &search_set);
    }
    pthread_setaffinity_np(search_thread.native_handle(),
                           sizeof(cpu_set_t), &search_set);
    for (int sub_idx : cached_indices) {
        cache_.mark_in_use(sub_idx);
        batch_ready_queue.push(sub_idx);
        // std::cout << "push cached_indices: " << sub_idx << std::endl;
    }
    for (int sub_idx : cached_not_deserialize_indices) {
        cache_.mark_in_use(sub_idx);
        batch_deserialize_queue.push(sub_idx);
    }
    for (int sub_idx : uncached_indices) {
        batch_fetch_queue.push(sub_idx);
        // std::cout << "push uncached_indices: " << sub_idx << std::endl;
    }

    batch_fetch_queue.push(-1);

   
    {
        std::unique_lock<std::mutex> lock(batch_done_mtx);
        bool completed = batch_done_cv.wait_for(
            lock,
            std::chrono::seconds(10),
            [&]() { return batch_tasks_remaining.load() == 0; }
        );

        if (!completed) {
            std::cerr << "Batch timeout - forcing completion" << std::endl;
            batch_fetch_queue.stop();
            batch_ready_queue.stop();
            batch_deserialize_queue.stop();
        }
    }

    // By now, tasks_remaining == 0 exactly once.  Both threads must have seen their single "−1":
    if (fetch_thread.joinable())  fetch_thread.join();
    if (search_thread.joinable()) search_thread.join();
    if (deserialize_thread.joinable()) deserialize_thread.join();
    cache_.clear_usage_tracking();  // Thread-safe version (fixes race condition)
    // std::cout << "sub_search_pipelined done" << std::endl;
    return std::make_tuple(total_compute_time, total_network_latency, total_deserialize_time);
}


std::vector<uint8_t> LocalHnsw::fetch_sub_hnsw_pipelined(int sub_idx) {
    // Validate sub_idx
    if (sub_idx < 0 || sub_idx >= num_sub_hnsw) {
        std::cerr << "Invalid sub_idx: " << sub_idx << std::endl;
        return std::vector<uint8_t>();
    }

    // Calculate positions and length under shared lock
    uint64_t base_offset = current_rdma_base_offset_.load(std::memory_order_acquire);
    uint64_t rel_start, rel_end;
    {
        std::shared_lock<std::shared_mutex> lock(epoch_mutex_);
        if (static_cast<size_t>(sub_idx * 2 + 1) >= offset_subhnsw_.size()) {
            std::cerr << "Error: offset_subhnsw_ out of bounds in fetch for sub_idx " << sub_idx << std::endl;
            return std::vector<uint8_t>();
        }
        rel_start = offset_subhnsw_[sub_idx * 2];
        rel_end = offset_subhnsw_[sub_idx * 2 + 1];
    }
    uint64_t start_pos = base_offset + rel_start;
    u32 length = static_cast<u32>(rel_end - rel_start);

    // std::cout << "Start pos: " << start_pos << " (base=" << base_offset << ", rel=" << rel_start << ")" << std::endl;
    // std::cout << "Length: " << length << std::endl;

    // Access local memory
    if (!local_mem_) {
        std::cerr << "Error: local_mem_ is null" << std::endl;
        return std::vector<uint8_t>();
    }

    if (!local_mem_->raw_ptr) {
        std::cerr << "Error: local_mem_->raw_ptr is null" << std::endl;
        return std::vector<uint8_t>();
    }

    uint8_t* recv_buffer = reinterpret_cast<uint8_t*>(local_mem_->raw_ptr);
    size_t local_mem_size = local_mem_->sz;

    // std::cout << "Local mem size: " << local_mem_size << std::endl;

    // Ensure the local buffer is large enough
    if (length > local_mem_size) {
        std::cerr << "Data length exceeds local buffer size" << std::endl;
        return std::vector<uint8_t>();
    }

    // Perform RDMA read to fetch the data
    auto res_s = qp->send_normal(
        {
            .op = IBV_WR_RDMA_READ,
            .flags = IBV_SEND_SIGNALED,
            .len = length,
            .wr_id = 0
        },
        {
            .local_addr = reinterpret_cast<RMem::raw_ptr_t>(recv_buffer),
            .remote_addr = start_pos,
            .imm_data = 0
        }
    );

    RDMA_ASSERT(res_s == IOCode::Ok) << "RDMA read failed: " << res_s.desc;
    // std::cout << "RDMA read initiated for sub_idx " << sub_idx << std::endl;

    // Wait for completion
    auto res_p = qp->wait_one_comp();
    RDMA_ASSERT(res_p == IOCode::Ok) << "RDMA read completion failed " << std::endl;
    // std::cout << "RDMA read completed for sub_idx " << sub_idx << std::endl;

    // Deserialize the sub-index
    std::vector<uint8_t> serialized_data(recv_buffer, recv_buffer + length);

    // Optionally clear the buffer
    memset(recv_buffer, 0, length);

    return serialized_data;
}

std::pair<bool, double> LocalHnsw::fetch_sub_hnsw_and_put_pipelined(int sub_idx, ThreadSafeQueue<int>& batch_deserialize_queue) {
    double fetch_duration_us = 0.0;
    // Validate sub_idx
    if (sub_idx < 0 || sub_idx >= num_sub_hnsw) {
        std::cerr << "Invalid sub_idx: " << sub_idx << std::endl;
        return {false, fetch_duration_us};
    }

    // Calculate positions and length under shared lock
    // (init() can be called from main thread while worker threads are fetching)
    uint64_t base_offset = current_rdma_base_offset_.load(std::memory_order_acquire);
    uint64_t rel_start, rel_end;
    {
        std::shared_lock<std::shared_mutex> lock(epoch_mutex_);
        if (static_cast<size_t>(sub_idx * 2 + 1) >= offset_subhnsw_.size()) {
            std::cerr << "Error: offset_subhnsw_ out of bounds in fetch_and_put for sub_idx " << sub_idx << std::endl;
            return {false, fetch_duration_us};
        }
        rel_start = offset_subhnsw_[sub_idx * 2];
        rel_end = offset_subhnsw_[sub_idx * 2 + 1];
    }
    uint64_t start_pos = base_offset + rel_start;
    u32 length = static_cast<u32>(rel_end - rel_start);

    // std::cout << "Start pos: " << start_pos << " (base=" << base_offset << ", rel=" << rel_start << ")" << std::endl;
    // std::cout << "End pos: " << end_pos << std::endl;
    // std::cout << "Length: " << length << std::endl;

    // Access local memory
    if (!local_mem_) {
        std::cerr << "Error: local_mem_ is null" << std::endl;
        return {false, fetch_duration_us};
    }

    if (!local_mem_->raw_ptr) {
        std::cerr << "Error: local_mem_->raw_ptr is null" << std::endl;
        return {false, fetch_duration_us};
    }

    uint8_t* recv_buffer = reinterpret_cast<uint8_t*>(local_mem_->raw_ptr);
    size_t local_mem_size = local_mem_->sz;

    // std::cout << "Local mem size: " << local_mem_size << std::endl;

    // Ensure the local buffer is large enough
    if (length > local_mem_size) {
        std::cerr << "Data length exceeds local buffer size" << std::endl;
        return {false, fetch_duration_us};
    }

    // Perform RDMA read to fetch the data
    auto fetch_start = std::chrono::high_resolution_clock::now();
    auto res_s = qp->send_normal(
        {
            .op = IBV_WR_RDMA_READ,
            .flags = IBV_SEND_SIGNALED,
            .len = length,
            .wr_id = 0
        },
        {
            .local_addr = reinterpret_cast<RMem::raw_ptr_t>(recv_buffer),
            .remote_addr = start_pos,
            .imm_data = 0
        }
    );

    RDMA_ASSERT(res_s == IOCode::Ok) << "RDMA read failed: " << res_s.desc;
    // std::cout << "RDMA read initiated for sub_idx " << sub_idx << std::endl;

    // Wait for completion
    auto res_p = qp->wait_one_comp();
    RDMA_ASSERT(res_p == IOCode::Ok) << "RDMA read completion failed " << std::endl;
    // std::cout << "First 8 bytes of recv_buffer for sub_idx " << sub_idx << ": ";
    // for (int i = 0; i < std::min((u32)8, length); ++i) {
    //     printf("%02X ", recv_buffer[i]);
    // }
    // std::cout << std::endl;
    // std::cout << "RDMA read completed for sub_idx " << sub_idx << std::endl;
    auto fetch_end = std::chrono::high_resolution_clock::now();
    fetch_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(fetch_end - fetch_start).count();
    // Deserialize the sub-index
    std::vector<uint8_t> serialized_data(recv_buffer, recv_buffer + length);
    if (cache_.put(sub_idx, std::move(serialized_data))) {
        batch_deserialize_queue.push(sub_idx);
        return {true, fetch_duration_us};
    } else {
        std::cerr << "Failed to put index " << sub_idx << " in cache" << std::endl;
        return {false, fetch_duration_us};
    }
    // Optionally clear the buffer
    // memset(recv_buffer, 0, length);

    return {true, fetch_duration_us};
}

// std::pair<bool, double> LocalHnsw::fetch_sub_hnsw_batch_with_doorbell_and_put_pipelined(const std::vector<int>& sub_indices, ThreadSafeQueue<int>& batch_deserialize_queue, bool& stop) {
//     size_t batch_size = sub_indices.size();
//     double fetch_duration_us = 0.0;
//     auto fetch_start = std::chrono::high_resolution_clock::now();
//     // Prepare vectors for RDMA operations
//     std::vector<uint64_t> sizes(batch_size);
//     std::vector<uint64_t> local_offsets(batch_size);
//     std::vector<uint64_t> remote_offsets(batch_size);
//     size_t total_size = 0;

//     // Determine sizes and offsets; note that remote_offsets are relative offsets stored in your "offset" array.
//     for (size_t i = 0; i < batch_size; ++i) {
//         int sub_idx = sub_indices[i];
//         size_t index_start = offset[sub_idx * 2 + 0];
//         size_t index_end   = offset[sub_idx * 2 + 1];
//         size_t size        = index_end - index_start;
//         sizes[i] = size;
//         remote_offsets[i] = index_start; // relative offset as stored
//         local_offsets[i]  = total_size;
//         total_size += size;
//     }

//     if (!local_mem_ || !local_mem_->raw_ptr) {
//         std::cerr << "Error: local memory not properly allocated." << std::endl;
//         return {};
//     }
//     uint8_t* recv_buffer = reinterpret_cast<uint8_t*>(local_mem_->raw_ptr);
//     size_t local_mem_size = local_mem_->sz;
//     if (total_size > local_mem_size) {
//         RDMA_LOG(ERROR) << "Local MR size insufficient for batch fetch. Required size: " << total_size;
//         return {};
//     }

//     // Check against the maximum allowed batch size.
//     const int max_batch_size = 16;  
//     if (batch_size > max_batch_size) {
//         std::cerr << "Batch size exceeds the maximum allowed batch size." << std::endl;
//         return {};
//     }

//     // Prepare the batched RDMA read operations.
//     BenchOp<> ops[max_batch_size];
//     u32 lkey = local_mr->get_reg_attr().value().key;
//     u32 rkey = remote_attr.key;

//     for (int i = 0; i < batch_size; ++i) {
//         ops[i].set_type(0);  // RDMA_READ
//         // Compute the absolute remote address by adding the base remote address.
//         uint64_t remote_addr = reinterpret_cast<uint64_t>(remote_attr.buf) + remote_offsets[i];
//         // Note: init_rbuf expects a pointer to the remote memory address.
//         ops[i].init_rbuf(reinterpret_cast<u64*>(remote_addr), rkey);
//         ops[i].init_lbuf(reinterpret_cast<u64*>(recv_buffer + local_offsets[i]), sizes[i], lkey);
//         ops[i].set_wrid(i);
//         ops[i].set_flags(0);  // Clear any flags.
//         if (i != 0) {
//             ops[i - 1].set_next(&ops[i]);  // Chain the operations.
//         }
//     }
//     // Only the last op is signaled to reduce overhead.
//     ops[batch_size - 1].set_flags(IBV_SEND_SIGNALED);

//     // Execute the entire batch with one doorbell ring.
//     auto res_s = ops[0].execute_batch(qp_shared);
//     if (res_s != IOCode::Ok) {
//         RDMA_LOG(ERROR) << "Failed to execute RDMA read batch: " << res_s.desc;
//         return {};
//     }

//     // Wait for the batched operation to complete.
//     auto res_p = qp_shared->wait_one_comp();
//     if (res_p != IOCode::Ok) {
//         RDMA_LOG(ERROR) << "Failed to wait for RDMA completion." << std::endl;
//         return {};
//     }

//     auto fetch_end = std::chrono::high_resolution_clock::now();
//     fetch_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(fetch_end - fetch_start).count();

//     // Deserialize the sub-indices in parallel.
//     std::atomic<bool> error_occurred(false);
//     for (size_t i = 0; i < batch_size; ++i) {
//         uint8_t* data_ptr = recv_buffer + local_offsets[i];
//         size_t data_size = sizes[i];
//         std::vector<uint8_t> serialized_data(data_ptr, data_ptr + data_size);
//         if (cache_.put(sub_indices[i], std::move(serialized_data))) {
//             batch_deserialize_queue.push(sub_indices[i]);
//         } else {
//             std::cerr << "Failed to put index " << sub_indices[i] << " in cache" << std::endl;
//             error_occurred.store(true, std::memory_order_relaxed);
//         }
//     }
//     if (error_occurred.load(std::memory_order_relaxed)) {
//         return {false, 0.0};
//     }
//     if (stop) {
//        batch_deserialize_queue.push(-1); 
//     }
//     // Optionally clear only the used portion of the buffer.
//     memset(recv_buffer, 0, total_size);

//     return {true, fetch_duration_us};
// }

// TODO: merge offesets
std::pair<bool, double> LocalHnsw::fetch_sub_hnsw_batch_with_doorbell_and_put_pipelined(
        const std::vector<int>& sub_indices,
        ThreadSafeQueue<int>& batch_deserialize_queue,
        bool& stop) {

    const size_t batch_size = sub_indices.size();
    if (batch_size == 0) {
        return {false, 0.0};
    }

    std::vector<uint64_t> sizes(batch_size);
    std::vector<uint64_t> local_offsets(batch_size);
    std::vector<uint64_t> remote_offsets(batch_size);
    size_t total_size = 0;

    // Get base offset for this epoch
    uint64_t base_offset = current_rdma_base_offset_.load(std::memory_order_acquire);

    // Read offsets under shared lock
    {
        std::shared_lock<std::shared_mutex> lock(epoch_mutex_);
        for (size_t i = 0; i < batch_size; ++i) {
            int sub_idx = sub_indices[i];
            if (static_cast<size_t>(sub_idx * 2 + 1) >= offset_subhnsw_.size()) {
                std::cerr << "Error: offset_subhnsw_ out of bounds in fetch_batch_doorbell_put for sub_idx " << sub_idx << std::endl;
                return {false, 0.0};
            }
            uint64_t rel_start = offset_subhnsw_[sub_idx * 2 + 0];
            uint64_t rel_end   = offset_subhnsw_[sub_idx * 2 + 1];
            uint64_t sz        = rel_end - rel_start;
            sizes[i]           = sz;
            remote_offsets[i]  = base_offset + rel_start;
            local_offsets[i]   = total_size;    
            total_size        += sz;
        }
    }

    if (!local_mem_ || !local_mem_->raw_ptr) {
        std::cerr << "[Error] Local MR null\n";
        return {false, 0.0};
    }
    uint8_t* recv_buffer = reinterpret_cast<uint8_t*>(local_mem_->raw_ptr);
    const size_t local_mr_size = local_mem_->sz;
    if (total_size > local_mr_size) {
        RDMA_LOG(ERROR) << "Local MR too small, required: " << total_size << " bytes, only: " << local_mr_size;
        return {false, 0.0};
    }

    constexpr int MAX_BATCH = 16;
    if (batch_size > MAX_BATCH) {
        std::cerr << "[Error] Batch size (" << batch_size << ") > MAX_BATCH (" << MAX_BATCH << ")\n";
        return {false, 0.0};
    }

    BenchOp<> ops[MAX_BATCH];
    u32 lkey = local_mr->get_reg_attr().value().key;
    u32 rkey = remote_attr.key;
    uint64_t remote_base = reinterpret_cast<uint64_t>(remote_attr.buf);

    for (int i = 0; i < (int)batch_size; ++i) {
        ops[i].set_type(0); // IBV_WR_RDMA_READ
        uint64_t abs_remote = remote_base + remote_offsets[i];
        ops[i].init_rbuf(reinterpret_cast<u64*>(abs_remote), rkey);
        ops[i].init_lbuf(reinterpret_cast<u64*>(recv_buffer + local_offsets[i]), sizes[i], lkey);
        ops[i].set_wrid(i);
        if (i + 1 < (int)batch_size) {
        ops[i].set_flags(0); // unsignaled
        ops[i].set_next(&ops[i+1]);
        } else {
            ops[i].set_flags(IBV_SEND_SIGNALED);
        } 
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    auto rc_s = ops[0].execute_batch(qp_shared);
    if (rc_s != IOCode::Ok) {
        RDMA_LOG(ERROR) << "Doorbell fetch 失败: " << rc_s.desc;
        return {false, 0.0};
    }
    auto rc_p = qp_shared->wait_one_comp();
    if (rc_p != IOCode::Ok) {
        RDMA_LOG(ERROR) << "Doorbell completion 失败: " ;
        return {false, 0.0};
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double fetch_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

    std::atomic<bool> error_flag(false);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)batch_size; ++i) {
        size_t sz    = sizes[i];
        uint8_t* ptr = recv_buffer + local_offsets[i];

        std::vector<uint8_t> serialized_data;
        serialized_data.resize(sz);
        std::memcpy(serialized_data.data(), ptr, sz);

        bool ok = cache_.put(sub_indices[i], std::move(serialized_data));
        if (!ok) {
            std::cerr << "[Error] cache_.put fail, sub_idx=" << sub_indices[i] << "\n";
            error_flag.store(true, std::memory_order_relaxed);
        } else {
            batch_deserialize_queue.push(sub_indices[i]);
        }
    }

    if (error_flag.load(std::memory_order_relaxed)) {
        return {false, 0.0};
    }

    if (stop) {
        batch_deserialize_queue.push(-1);
    }


    return {true, fetch_duration_us};
}


PipelinedSearchManager::PipelinedSearchManager(LocalHnsw* parent, int core_start, int cores_per_worker)
    : parent_hnsw_(parent) {
    
    // Launch the persistent worker threads
    fetch_thread_ = std::thread(&PipelinedSearchManager::fetch_worker, this);
    deserialize_thread_ = std::thread(&PipelinedSearchManager::deserialize_worker, this);
    search_thread_ = std::thread(&PipelinedSearchManager::search_worker, this);

    // --- Pin threads to specific cores (affinity) ---
    cpu_set_t fetch_set, deserialize_set, search_set;
    CPU_ZERO(&fetch_set);
    CPU_ZERO(&deserialize_set);
    CPU_ZERO(&search_set);

    // Pin fetch_thread to the first core
    CPU_SET(core_start, &fetch_set);
    if (pthread_setaffinity_np(fetch_thread_.native_handle(), sizeof(cpu_set_t), &fetch_set) != 0) {
        std::cerr << "Error setting affinity for fetch thread" << std::endl;
    }

    // Pin deserialize_thread to the second core
    CPU_SET(core_start + 1 * 2, &deserialize_set); // Simplified for clarity
     if (pthread_setaffinity_np(deserialize_thread_.native_handle(), sizeof(cpu_set_t), &deserialize_set) != 0) {
        std::cerr << "Error setting affinity for deserialize thread" << std::endl;
    }

    // Pin search_thread to the remaining cores
    for (int i = 2; i < cores_per_worker; i++) {
        CPU_SET(core_start + i * 2, &search_set);
    }
    if (pthread_setaffinity_np(search_thread_.native_handle(), sizeof(cpu_set_t), &search_set) != 0) {
        std::cerr << "Error setting affinity for search thread" << std::endl;
    }
}

PipelinedSearchManager::~PipelinedSearchManager() {
    // Signal threads to stop
    stop_threads_.store(true);

    // Unblock any threads waiting on queues
    batch_fetch_queue_.stop();
    batch_deserialize_queue_.stop();
    batch_ready_queue_.stop();

    // Wait for threads to finish
    if (fetch_thread_.joinable()) fetch_thread_.join();
    if (deserialize_thread_.joinable()) deserialize_thread_.join();
    if (search_thread_.joinable()) search_thread_.join();
}

std::tuple<double, double, double> PipelinedSearchManager::process_batch(
    const int n,
    const float* query,
    int K_sub,
    float* distances,
    dhnsw_idx_t* labels,
    std::unordered_map<int, std::unordered_set<int>>& searchset,
    dhnsw_idx_t* sub_hnsw_tags,
    int ef)
{
    // Use a lock to ensure that only one batch is processed at a time by this manager instance
    std::lock_guard<std::mutex> lock(processing_mutex_);

    // --- 1. Reset and Setup State for the New Batch ---
    auto& cache_ = parent_hnsw_->cache_; // Get cache from parent
    this->n_ = n;
    this->batch_queries_ptr_ = query;
    this->batch_distances_ptr_ = distances;
    this->batch_labels_ptr_ = labels;
    this->batch_tags_ptr_ = sub_hnsw_tags;
    this->searchset_ptr_ = &searchset;
    this->batch_dim_ = parent_hnsw_->d;
    this->batch_efSearch_ = ef;
    this->batch_K_sub_ = K_sub;

    total_compute_time_ = 0.0;
    total_network_latency_ = 0.0;
    total_deserialize_time_ = 0.0;
    
    std::fill(distances, distances + (size_t)n * K_sub, std::numeric_limits<float>::max());
    std::fill(labels, labels + (size_t)n * K_sub, -1);
    std::fill(sub_hnsw_tags, sub_hnsw_tags + (size_t)n * K_sub, -1);
    
    int total_shards = (int)searchset.size();
    batch_tasks_remaining_.store(total_shards);

    if (total_shards == 0) {
        return {0.0, 0.0, 0.0};
    }

    // --- 2. Triage work and push to queues ---
    std::vector<int> cached_indices;
    std::vector<int> uncached_indices;
    std::vector<int> cached_not_deserialized_indices;

    for (auto const& [sub_idx, _] : searchset) {
        if (cache_.has_index(sub_idx)) {
            cached_indices.push_back(sub_idx);
        } else if (cache_.has_serialized_not_index(sub_idx)) {
            cached_not_deserialized_indices.push_back(sub_idx);
        } else {
            uncached_indices.push_back(sub_idx);
        }
    }

    for (auto const &kv : cache_.cache_map_) {
        int sub_idx = kv.first;
        if (searchset.count(sub_idx) == 0) {
            cache_.mark_used(sub_idx);  // Thread-safe version (fixes race condition)
        }
    }
      for (int sub_idx : cached_indices) {
        cache_.mark_in_use(sub_idx);
        batch_ready_queue_.push(sub_idx);
        // std::cout << "push cached_indices: " << sub_idx << std::endl;
    }
    for (int sub_idx : cached_not_deserialized_indices) {
        cache_.mark_in_use(sub_idx);
        batch_deserialize_queue_.push(sub_idx);
    }
    for (int sub_idx : uncached_indices) {
        batch_fetch_queue_.push(sub_idx);
        // std::cout << "push uncached_indices: " << sub_idx << std::endl;
    }


    // The -1 sentinel now means "end of items for this batch" for the fetcher
    batch_fetch_queue_.push(-1);
    // batch_deserialize_queue_.push(-1);
    // batch_ready_queue_.push(-1);
    // --- 3. Wait for this batch to complete ---
    {
        std::unique_lock<std::mutex> lock(batch_done_mtx_);
        if (!batch_done_cv_.wait_for(lock, std::chrono::seconds(60), [&]{ return batch_tasks_remaining_.load() == 0; })) {
             std::cerr << "Batch timeout! " << batch_tasks_remaining_.load() << "/" << total_shards << " tasks remaining." << std::endl;
             // Handle timeout if necessary, but don't stop the persistent threads
        }
    }
    
    parent_hnsw_->cache_.clear_usage_tracking();  // Thread-safe version (fixes race condition)

    return std::make_tuple(total_compute_time_.load(), total_network_latency_.load(), total_deserialize_time_.load());
}


// --- Worker Implementations ---

void PipelinedSearchManager::fetch_worker() {
    std::cout << "fetch_worker start" << std::endl;
    omp_set_num_threads(1);
    while (!stop_threads_) {
        int sub_idx = batch_fetch_queue_.pop(); // Blocks here until work is available or stopped
        if (sub_idx < 0) { // Sentinel for end of this batch's fetch list
            batch_deserialize_queue_.push(-1); // Pass sentinel to next stage
            if (stop_threads_) break;
            continue; 
        }

        // Check if overflow was detected - skip RDMA fetch if so
        if (parent_hnsw_->has_overflow_detected()) {
            std::cerr << "[FETCH] Overflow detected, skipping RDMA fetch for sub_idx " << sub_idx << std::endl;
            // Decrement batch_tasks_remaining_ since we're skipping this task
            int prev = batch_tasks_remaining_.fetch_sub(1);
            if (prev == 1) {
                std::lock_guard<std::mutex> lk(batch_done_mtx_);
                batch_done_cv_.notify_all();
            }
            continue;
        }

        auto& cache_ = parent_hnsw_->cache_;
        int items = parent_hnsw_->cache_.capacity_; 
        for (int i = 0; i < items; ++i) {
            while (!cache_.evict_one()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        std::pair<bool, double> fetch_result;
        // int available = cache_.capacity_ - cache_.cache_map_.size();

        //     if(cache_.cache_map_.size() <= 1 || available == 1){
                auto fetch_trace_start = std::chrono::high_resolution_clock::now();
                fetch_result = parent_hnsw_->fetch_sub_hnsw_and_put_pipelined(sub_idx, batch_deserialize_queue_);
                auto fetch_trace_end = std::chrono::high_resolution_clock::now();
                // g_tracer.add_event({
                //     "Fetch", 
                //     {sub_idx}, 
                //     std::chrono::duration_cast<std::chrono::microseconds>(fetch_trace_start - pipeline_start_time).count(),
                //     std::chrono::duration_cast<std::chrono::microseconds>(fetch_trace_end - pipeline_start_time).count()
                // });
            // }
            // else{
            //     //pop capacity - size and doorbell fetch
            //     bool stop = false;
            //     std::vector<int> sub_indices;
            //     sub_indices.push_back(sub_idx);
            //     for (int i = 1; i < available; i++) {
            //         int tmp = batch_fetch_queue_.pop();
            //         if (tmp < 0) {
            //             stop = true;
            //             break;
            //         }
            //         sub_indices.push_back(tmp);
            //     }
            //     auto fetch_trace_start = std::chrono::high_resolution_clock::now();
            //     fetch_result = parent_hnsw_->fetch_sub_hnsw_batch_with_doorbell_and_put_pipelined(sub_indices, batch_deserialize_queue_, stop);
            //     auto fetch_trace_end = std::chrono::high_resolution_clock::now();
            //     // g_tracer.add_event({
            //     //     "Fetch", 
            //     //     sub_indices, 
            //     //     std::chrono::duration_cast<std::chrono::microseconds>(fetch_trace_start - pipeline_start_time).count(),
            //     //     std::chrono::duration_cast<std::chrono::microseconds>(fetch_trace_end - pipeline_start_time).count()
            //     // });
            // }
            // faiss::IndexHNSWFlat* idx = DistributedHnsw::deserialize_sub_hnsw(serialized_data);
          
            // std::cout << "fetch_duration: " << fetch_duration << std::endl;
            // std::cout << "fetch done for sub_idx: " << sub_idx << std::endl;

            if (!fetch_result.first) {
                std::cerr << "Failed to put index " << sub_idx << " in cache" << std::endl;
            }
            double old_val = total_network_latency_.load();
            while (!total_network_latency_.compare_exchange_weak(old_val, old_val + fetch_result.second));  

    }
}

void PipelinedSearchManager::deserialize_worker() {
    std::cout << "deserialize_worker start" << std::endl;
    omp_set_num_threads(1);
    while (!stop_threads_) {
        int sub_idx = batch_deserialize_queue_.pop();
        if (sub_idx < 0) { // Sentinel for end of this batch's deserialize list
            batch_ready_queue_.push(-1); // Pass sentinel to the searcher
            if (stop_threads_) break;
            continue; 
        }
        
        // Check if overflow was detected - skip processing if so
        if (parent_hnsw_->has_overflow_detected()) {
            std::cerr << "[DESERIALIZE] Overflow detected, skipping sub_idx " << sub_idx << std::endl;
            // Decrement batch_tasks_remaining_ since we're skipping this task
            int prev = batch_tasks_remaining_.fetch_sub(1);
            if (prev == 1) {
                std::lock_guard<std::mutex> lk(batch_done_mtx_);
                batch_done_cv_.notify_all();
            }
            continue;
        }
        
        auto& cache_ = parent_hnsw_->cache_;
        auto serialized_and_index = cache_.get_serialized_and_index(sub_idx);
        if (!serialized_and_index.first) {
            std::cerr << "[CRITICAL] Deserialize worker: sub_idx " << sub_idx 
                      << " not found in cache or returned null data ptr. Skipping." << std::endl;
        } else if (serialized_and_index.second == nullptr) { // Needs deserialization
            if (serialized_and_index.first->empty()) {
                 std::cerr << "[WARNING] Deserialize worker: sub_idx " << sub_idx 
                           << " has an empty data vector. Cannot deserialize." << std::endl;
            } else {
                // Wrap deserialization in try-catch to handle overflow-related IO errors
                try {
                    auto start = std::chrono::high_resolution_clock::now();
                    faiss::IndexHNSWFlat* idx_tmp = parent_hnsw_->deserialize_sub_hnsw_pipelined_(*serialized_and_index.first,sub_idx);
                    auto end = std::chrono::high_resolution_clock::now();
                    double old_val = total_deserialize_time_.load();
                    while (!total_deserialize_time_.compare_exchange_weak(old_val, old_val + std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()));
                    cache_.update(sub_idx, std::shared_ptr<faiss::IndexHNSWFlat>(idx_tmp));
                } catch (const std::runtime_error& e) {
                    // Check if this is due to overflow - if so, handle gracefully
                    if (parent_hnsw_->has_overflow_detected()) {
                        std::cerr << "[DESERIALIZE] Deserialization failed for sub_idx " << sub_idx 
                                  << " due to overflow (error: " << e.what() << "). Skipping." << std::endl;
                    } else {
                        std::cerr << "[DESERIALIZE] Deserialization failed for sub_idx " << sub_idx 
                                  << " (error: " << e.what() << "). Re-throwing." << std::endl;
                        throw;  // Re-throw if not overflow-related
                    }
                }
            }
        }

        batch_ready_queue_.push(sub_idx);
    }
}


void PipelinedSearchManager::search_worker() {
    std::cout << "search_worker start" << std::endl;
    omp_set_num_threads(7); // Set desired parallelism for searching
    while (!stop_threads_) {
        int sub_idx = batch_ready_queue_.pop();
        if (sub_idx < 0) { // Sentinel for end of this batch's search list
            if (stop_threads_) break;
            continue; 
        }

        // Check if overflow was detected - skip search if so
        if (parent_hnsw_->has_overflow_detected()) {
            // Decrement remaining tasks since we're skipping this one
            int prev = batch_tasks_remaining_.fetch_sub(1);
            if (prev == 1) {
                std::lock_guard<std::mutex> lk(batch_done_mtx_);
                batch_done_cv_.notify_all();
            }
            continue;
        }

        auto& cache_ = parent_hnsw_->cache_;
        // Use atomic get+mark_in_use to prevent use-after-free race condition
        // where evict_one() could delete the index between get and mark_in_use
        std::shared_ptr<faiss::IndexHNSWFlat> idx = cache_.get_index_ptr_and_mark_in_use(sub_idx);
        auto it = searchset_ptr_->find(sub_idx);
        if (it == searchset_ptr_->end()) {
            std::cerr << "[CRITICAL] Segfault averted: sub_idx " << sub_idx 
                      << " was in ready_queue but not in the current batch's searchset!" << std::endl;
            cache_.unmark_in_use(sub_idx);  // Clean up since we're not searching
            // Must decrement task counter to avoid hanging
            int prev = batch_tasks_remaining_.fetch_sub(1);
            if (prev == 1) {
                std::lock_guard<std::mutex> lk(batch_done_mtx_);
                batch_done_cv_.notify_all();
            }
        } else if (idx == nullptr) {
            std::cerr << "[WARNING] Search worker: sub_hnsw " << sub_idx << " has null index ptr in cache." << std::endl;
            // Must decrement task counter to avoid hanging (can happen during overflow)
            int prev = batch_tasks_remaining_.fetch_sub(1);
            if (prev == 1) {
                std::lock_guard<std::mutex> lk(batch_done_mtx_);
                batch_done_cv_.notify_all();
            }
        } else {
            auto search_start = std::chrono::high_resolution_clock::now();
            
            cache_.mark_used(sub_idx);  // Thread-safe version (fixes race condition)
            // Look up which queries go to this shard         
            auto it = searchset_ptr_->find(sub_idx);

            const std::unordered_set<int>& qset = it->second;
            int m = (int)qset.size();

            // Build a local "subQ" of size m × dim
            std::vector<float> subQ((size_t)m * batch_dim_);
            {
                int i = 0;
                for (int qi : qset) {
                    const float* src = batch_queries_ptr_ + (size_t)qi * batch_dim_;
                    float* dst       = subQ.data() + (size_t)i * batch_dim_;
                    std::memcpy(dst, src, sizeof(float) * batch_dim_);
                    ++i;
                }
            }

            // Run the actual HNSW search on “m” queries
            std::vector<float> tmpd((size_t)m * batch_K_sub_);
            std::vector<dhnsw_idx_t> tmpl((size_t)m * batch_K_sub_);

            idx->hnsw.efSearch = batch_efSearch_;
            idx->search(m, subQ.data(), batch_K_sub_, tmpd.data(), tmpl.data());
            cache_.unmark_in_use(sub_idx);
            // std::cout << "search done for sub_idx: " << sub_idx << std::endl;
            // Merge results back into “distances/labels/sub_hnsw_tags”
            {
                std::lock_guard<std::mutex> lg(batch_done_mtx_);
                int i = 0;
                for (int qi : qset) {
                    float*    outd = batch_distances_ptr_ + (size_t)qi * batch_K_sub_;
                    dhnsw_idx_t* outl = batch_labels_ptr_ + (size_t)qi * batch_K_sub_;
                    dhnsw_idx_t* outt = batch_tags_ptr_ + (size_t)qi * batch_K_sub_;

                    for (int k = 0; k < batch_K_sub_; ++k) {
                        float d = tmpd[(size_t)i * batch_K_sub_ + k];
                        if (d < outd[batch_K_sub_ - 1]) {
                            int pos = batch_K_sub_ - 1;
                            while (pos > 0 && outd[pos - 1] > d) {
                                outd[pos]  = outd[pos - 1];
                                outl[pos]  = outl[pos - 1];
                                outt[pos]  = outt[pos - 1];
                                --pos;
                            }
                            outd[pos] = d;
                            outl[pos] = tmpl[(size_t)i * batch_K_sub_ + k];
                            outt[pos] = sub_idx;
                        }
                    }
                    ++i;
                }

                int rem = --batch_tasks_remaining_;
                if (rem == 0) {
                    batch_done_cv_.notify_one();
                }
            }
            auto search_end = std::chrono::high_resolution_clock::now();
            // g_tracer.add_event({
            //         "search", 
            //         {sub_idx}, 
            //         std::chrono::duration_cast<std::chrono::microseconds>(search_start - pipeline_start_time).count(),
            //         std::chrono::duration_cast<std::chrono::microseconds>(search_end - pipeline_start_time).count()
            //     });     
            auto search_duration = std::chrono::duration_cast<std::chrono::microseconds>(search_end - search_start).count();
            // std::cout << "search_duration: " << search_duration << std::endl;
            double old_val = total_compute_time_.load();
            while (!total_compute_time_.compare_exchange_weak(old_val, old_val + search_duration));
        }
    }
}

std::pair<double, double> LocalHnsw::insert_to_server(const int n, std::vector<float>& data) {
    float* construct_distances = new float[n];
    dhnsw_idx_t* construct_labels = new dhnsw_idx_t[n];
    std::vector<int> sub_hnsw_toinsert; 
    this->meta_search(n, data.data(), 1, construct_distances, construct_labels, sub_hnsw_toinsert);
    delete[] construct_distances;
    delete[] construct_labels;
    std::unordered_map<int, std::vector<float>> insertset;
    for (int i = 0; i < n; i++) {
        int sub_idx = sub_hnsw_toinsert[i];
        insertset[sub_idx].insert(insertset[sub_idx].end(), data.begin() + i * d, data.begin() + (i + 1) * d);
    }
    // Separate sub_hnsw indices into cached and uncached
    std::vector<int> cached_sub_indices;
    std::vector<int> uncached_sub_indices;
    std::unordered_map<int, std::vector<float>> cached_insertset;
    std::unordered_map<int, std::vector<float>> uncached_insertset;

    for (const auto& entry : insertset) {
        int sub_idx = entry.first;
        const std::vector<float>& insert_data = entry.second;

        // Check if the sub_hnsw is in the cache
        if (insert_cache_.find(sub_idx) != insert_cache_.end()) {
            // Sub_hnsw is in cache
            cached_sub_indices.push_back(sub_idx);
            cached_insertset[sub_idx] = insert_data;
        } else {
            // Sub_hnsw not in cache
            uncached_sub_indices.push_back(sub_idx);
            uncached_insertset[sub_idx] = insert_data;
        }
    }
    
    // First process cached sub_hnsw indices
    for (int sub_idx : cached_sub_indices) {
        // Check overflow flag before processing each sub_idx
        if (has_overflow_detected()) {
            std::cerr << "[INSERT] Overflow detected, aborting insert_to_server early (cached)" << std::endl;
            return std::make_pair(0, 0);
        }
        // std::cout << "cached_sub_indices " << sub_idx << std::endl;
        const std::vector<float>& insert_data = cached_insertset[sub_idx];
        // Sub_hnsw is in cache
        std::shared_ptr<faiss::IndexHNSWFlat> sub_index = insert_cache_[sub_idx];
        // std::cout << "sub_index->ntotal " << sub_index->ntotal << std::endl;
        // std::cout << "sub_index->hnsw.entry_point " << sub_index->hnsw.entry_point << std::endl;
        // std::cout << "sub_index->hnsw.max_level " << sub_index->hnsw.max_level << std::endl;
        // std::cout << "sub_index->hnsw.neighbors.size() " << sub_index->hnsw.neighbors.size() << std::endl;
        // std::cout << "sub_index->hnsw.levels.size() " << sub_index->hnsw.levels.size() << std::endl;
        // std::cout << "sub_index->hnsw.offsets.size() " << sub_index->hnsw.offsets.size() << std::endl;
        // std::cout << "sub_index->storage->xb.size() " << static_cast<faiss::IndexFlat*>(sub_index->storage)->xb.size() << std::endl;
        prepare_and_commit_update(sub_idx, sub_index, insert_data);
        
        // Check if overflow was just detected
        if (has_overflow_detected()) {
            std::cerr << "[INSERT] Overflow detected after prepare_and_commit_update for sub_idx " << sub_idx << ", aborting" << std::endl;
            return std::make_pair(0, 0);
        }
        // std::cout << "sub_idx " << sub_idx << " insert success" << std::endl;
    }


    for (int sub_idx : uncached_sub_indices) {
        // Check overflow flag before processing each sub_idx
        if (has_overflow_detected()) {
            std::cerr << "[INSERT] Overflow detected, aborting insert_to_server early (uncached)" << std::endl;
            return std::make_pair(0, 0);
        }
        //  std::cout << "uncached_sub_indices " << sub_idx << std::endl;
        const std::vector<float>& insert_data = uncached_insertset[sub_idx];
        
        // Wrap fetch in try-catch to handle any IO errors during overflow transition
        std::shared_ptr<faiss::IndexHNSWFlat> sub_index;
        try {
            sub_index = fetch_sub_hnsw(sub_idx);
        } catch (const std::runtime_error& e) {
            if (has_overflow_detected()) {
                std::cerr << "[INSERT] Fetch failed for sub_idx " << sub_idx 
                          << " during overflow (error: " << e.what() << "), aborting" << std::endl;
                return std::make_pair(0, 0);
            }
            throw;  // Re-throw if not overflow-related
        }
        
        if (!sub_index) {
            std::cerr << "[INSERT] fetch_sub_hnsw returned null for sub_idx " << sub_idx << std::endl;
            continue;
        }
        // std::cout << "fetch success"<< std::endl;
        // std::cout << "sub_index->ntotal " << sub_index->ntotal << std::endl;
        // std::cout << "sub_index->hnsw.entry_point " << sub_index->hnsw.entry_point << std::endl;
        // std::cout << "sub_index->hnsw.max_level " << sub_index->hnsw.max_level << std::endl;
        // std::cout << "sub_index->hnsw.neighbors.size() " << sub_index->hnsw.neighbors.size() << std::endl;
        // std::cout << "sub_index->hnsw.levels.size() " << sub_index->hnsw.levels.size() << std::endl;
        // std::cout << "sub_index->hnsw.offsets.size() " << sub_index->hnsw.offsets.size() << std::endl;
        // std::cout << "sub_index->storage->xb.size() " << static_cast<faiss::IndexFlat*>(sub_index->storage)->xb.size() << std::endl;
        {
            std::lock_guard<std::mutex> lock(cache_mutex_);
            insert_cache_[sub_idx] = std::shared_ptr<faiss::IndexHNSWFlat>(sub_index);
            // Add to front of LRU order (most recently used)
            insert_cache_lru_order_.push_front(sub_idx);
            
            if(insert_cache_.size() > num_sub_hnsw/10){
                evict_sub_hnsw_from_insert_cache();
            }
        }
        prepare_and_commit_update(sub_idx, sub_index, insert_data);
        
        // Check if overflow was just detected
        if (has_overflow_detected()) {
            std::cerr << "[INSERT] Overflow detected after prepare_and_commit_update for sub_idx " << sub_idx << ", aborting" << std::endl;
            return std::make_pair(0, 0);
        }
        // std::cout << "sub_idx " << sub_idx << " insert success" << std::endl;
    } 
    
    return std::make_pair(0, 0);
}

void LocalHnsw::evict_sub_hnsw_from_insert_cache() {
    //LRU eviction - evict one least recently used entry
    // std::cout << "!!!!!!!!!! EVICTING CACHE! Current size: " << insert_cache_.size() 
    //           << ", LRU list size: " << insert_cache_lru_order_.size() 
    //           << " !!!!!!!!!! " << std::endl;
    if (insert_cache_.empty() || insert_cache_lru_order_.empty()) {
        return;
    }
    
    // Get the least recently used entry (back of the list)
    int lru_sub_idx = insert_cache_lru_order_.back();
    
    auto it = insert_cache_.find(lru_sub_idx);
    if (it == insert_cache_.end()) {
        insert_cache_lru_order_.pop_back();
        return;
    }

    insert_cache_.erase(it);
    insert_cache_lru_order_.pop_back();
    // std::cout << "!!!!!!!!!! EVICTION DONE! Evicted: " << lru_sub_idx 
    //           << ", New size: " << insert_cache_.size() 
    //           << " !!!!!!!!!! " << std::endl;
}

std::shared_ptr<faiss::IndexHNSWFlat> LocalHnsw::get_from_insert_cache(int sub_idx) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    auto it = insert_cache_.find(sub_idx);
    if (it == insert_cache_.end()) {
        return nullptr;
    }
    
    // Move to front of LRU order (most recently used)
    auto lru_it = std::find(insert_cache_lru_order_.begin(), insert_cache_lru_order_.end(), sub_idx);
    if (lru_it != insert_cache_lru_order_.end()) {
        insert_cache_lru_order_.erase(lru_it);
    }
    insert_cache_lru_order_.push_front(sub_idx);
    
    return it->second;
}

void LocalHnsw::prepare_and_commit_update(int sub_idx, std::shared_ptr<faiss::IndexHNSWFlat> sub_index,  const std::vector<float>& insert_data){
        if (!sub_index) {
        std::cerr << "Error: Null sub_index in prepare_and_commit_update" << std::endl;
        return;
        }
        
        if (insert_data.empty() || insert_data.size() % d != 0) {
            std::cerr << "Error: Invalid insert_data size: " << insert_data.size() << " (not divisible by d=" << d << ")" << std::endl;
            return;
        }
        auto ntotal_before = sub_index->ntotal;
        auto entry_point_before = sub_index->hnsw.entry_point;
        auto max_level_before = sub_index->hnsw.max_level;
        auto neighbors_before = sub_index->hnsw.neighbors;
        auto* flat_storage = static_cast<faiss::IndexFlat*>(sub_index->storage);
        auto xb_before = flat_storage->xb.size();
        auto offsets_before = sub_index->hnsw.offsets.size();
        auto levels_before = sub_index->hnsw.levels.size(); 
        // auto neighbors_before = sub_index->hnsw.neighbors.size();
        // std::cout << "sub_index->add start" << std::endl;
        sub_index->add( insert_data.size()/d,insert_data.data());
        // std::cout << "sub_index->add end" << std::endl;
        // check where change, for levels,xb,offsets just append, if internal gap is full, using shared gap
        UpdateCommitRecord commit_record;
        commit_record.sub_index_id = sub_idx;
        if (ntotal_before != sub_index->ntotal) {
            commit_record.overwrites.push_back(OverwriteChange{
                offset_para_[sub_idx * 9 + 0],
                std::variant<faiss::Index::idx_t, faiss::storage_idx_t, size_t>{static_cast<faiss::Index::idx_t>(sub_index->ntotal)}
            });
            commit_record.overwrites.push_back(OverwriteChange{
                offset_para_[sub_idx * 9 + 6],
                std::variant<faiss::Index::idx_t, faiss::storage_idx_t, size_t>{static_cast<faiss::Index::idx_t>(sub_index->ntotal)}
        });
        }
        if (entry_point_before != sub_index->hnsw.entry_point) {
            commit_record.overwrites.push_back(OverwriteChange{
                offset_para_[sub_idx * 9 + 4],
                std::variant<faiss::Index::idx_t, faiss::storage_idx_t, size_t>{static_cast<faiss::storage_idx_t>(sub_index->hnsw.entry_point)}
            });
        }
        if (max_level_before != sub_index->hnsw.max_level) {
            commit_record.overwrites.push_back(OverwriteChange{
                offset_para_[sub_idx * 9 + 5],
                std::variant<faiss::Index::idx_t, faiss::storage_idx_t, size_t>{static_cast<size_t>(sub_index->hnsw.max_level)}
            });
        }
        
        if(sub_index->hnsw.levels.size() > levels_before){
            size_t total_new_levels = sub_index->hnsw.levels.size() - levels_before;
            const int* new_data_ptr = reinterpret_cast<const int*>(sub_index->hnsw.levels.data() + levels_before);
            uint64_t current_levels_end_addr = offset_para_[sub_idx * 9 + 1] + levels_before * sizeof(int) + sizeof(uint64_t);
            uint64_t available_levels_bytes = offset_para_[sub_idx * 9 + 2] - current_levels_end_addr;
            if(total_new_levels * sizeof(int) <= available_levels_bytes){
                commit_record.overwrites.push_back(OverwriteChange{
                offset_para_[sub_idx * 9 + 1],
                std::variant<faiss::Index::idx_t, faiss::storage_idx_t, size_t>{static_cast<size_t>(sub_index->hnsw.levels.size())}
            });
                commit_record.data_changes.emplace_back(
                    current_levels_end_addr,
                    new_data_ptr,
                    total_new_levels * sizeof(int)
                );
            }
            else{
                // Overflow detected - trigger reconstruction
                std::cout << "[OVERFLOW] levels overflow detected for sub_idx " << sub_idx << std::endl;
                overflow_detected_.store(true);
                last_overflow_type_ = "levels";
                last_overflow_sub_idx_ = sub_idx;
                if (overflow_callback_) {
                    overflow_callback_(sub_idx, "levels");
                }
                return;  // Cannot proceed without reconstruction
            }
        }
        if(sub_index->hnsw.offsets.size() > offsets_before){
            size_t total_new_offsets = sub_index->hnsw.offsets.size() - offsets_before;
            const size_t* new_data_ptr = reinterpret_cast<const size_t*>(sub_index->hnsw.offsets.data() + offsets_before);
            uint64_t current_offsets_end_addr = offset_para_[sub_idx * 9 + 2] + offsets_before * sizeof(size_t) + sizeof(uint64_t);
            uint64_t available_offsets_bytes = offset_para_[sub_idx * 9 + 3] - current_offsets_end_addr;
            if(total_new_offsets * sizeof(size_t) <= available_offsets_bytes){
                commit_record.overwrites.push_back(OverwriteChange{
                offset_para_[sub_idx * 9 + 2],
                std::variant<faiss::Index::idx_t, faiss::storage_idx_t, size_t>{static_cast<size_t>(sub_index->hnsw.offsets.size())}
            });
                commit_record.data_changes.emplace_back(
                    current_offsets_end_addr,
                    new_data_ptr,
                    total_new_offsets * sizeof(size_t)
                );
            }
            else{
                // Overflow detected - trigger reconstruction
                std::cout << "[OVERFLOW] offsets overflow detected for sub_idx " << sub_idx << std::endl;
                overflow_detected_.store(true);
                last_overflow_type_ = "offsets";
                last_overflow_sub_idx_ = sub_idx;
                if (overflow_callback_) {
                    overflow_callback_(sub_idx, "offsets");
                }
                return;  // Cannot proceed without reconstruction
            }
        }
        // std::cout << "sub_index->hnsw.neighbors.size() " << sub_index->hnsw.neighbors.size() << std::endl;
        // std::cout << "neighbors_before.size() " << neighbors_before.size() << std::endl;
        if(sub_index->hnsw.neighbors.size() > neighbors_before.size()){
        size_t search_limit = std::min(neighbors_before.size(), sub_index->hnsw.neighbors.size());
        size_t current_dirty_start = -1;
        for (size_t i = 0; i < search_limit; ++i) {
            if (sub_index->hnsw.neighbors[i] != neighbors_before[i]) {
                if (current_dirty_start == (size_t)-1) {
                    current_dirty_start = i; 
                }
            } else {
                if (current_dirty_start != (size_t)-1) {
                    size_t region_len = i - current_dirty_start;
                    commit_record.data_changes.push_back({
                        offset_para_[sub_idx * 9 + 3] + current_dirty_start * sizeof(faiss::storage_idx_t) + sizeof(uint64_t),
                        sub_index->hnsw.neighbors.data() + current_dirty_start,
                        region_len * sizeof(faiss::storage_idx_t)
                    });
                    current_dirty_start = -1;
                }
            }
        }

        if (current_dirty_start != (size_t)-1) {
            size_t region_len = search_limit - current_dirty_start;
            uint64_t current_neighbors_end_addr = offset_para_[sub_idx * 9 + 3] + current_dirty_start * sizeof(faiss::storage_idx_t) + sizeof(uint64_t);
            uint64_t available_neighbors_bytes = offset_para_[sub_idx * 9 + 4] - current_neighbors_end_addr;
            if(region_len * sizeof(faiss::storage_idx_t) <= available_neighbors_bytes){
            //     commit_record.overwrites.push_back(OverwriteChange{
            //     offset_para_[sub_idx * 9 + 3],
            //     std::variant<faiss::Index::idx_t, faiss::storage_idx_t, size_t>{static_cast<size_t>(sub_index->hnsw.neighbors.size())}
            // }); 
            commit_record.data_changes.push_back({
                offset_para_[sub_idx * 9 + 3] + current_dirty_start * sizeof(faiss::storage_idx_t) + sizeof(uint64_t),
                sub_index->hnsw.neighbors.data() + current_dirty_start,
                region_len * sizeof(faiss::storage_idx_t)
            });
            }
            else{
                // Overflow detected - trigger reconstruction
                std::cout << "[OVERFLOW] neighbors overflow detected for sub_idx " << sub_idx << std::endl;
                overflow_detected_.store(true);
                last_overflow_type_ = "neighbors";
                last_overflow_sub_idx_ = sub_idx;
                if (overflow_callback_) {
                    overflow_callback_(sub_idx, "neighbors");
                }
                return;  // Cannot proceed without reconstruction
            }
        }
        size_t append_count = sub_index->hnsw.neighbors.size() - neighbors_before.size();
        size_t append_size_bytes = append_count * sizeof(faiss::storage_idx_t);
        const faiss::storage_idx_t* append_data_ptr = sub_index->hnsw.neighbors.data() + neighbors_before.size();
        
        uint64_t append_target_addr = offset_para_[sub_idx * 9 + 3] + sizeof(uint64_t) + neighbors_before.size() * sizeof(faiss::storage_idx_t);
        
        uint64_t available_space = offset_para_[sub_idx * 9 + 4] - append_target_addr;
        if (append_size_bytes <= available_space) {
            commit_record.overwrites.push_back(OverwriteChange{
                offset_para_[sub_idx * 9 + 3],
                std::variant<faiss::Index::idx_t, faiss::storage_idx_t, size_t>{static_cast<size_t>(sub_index->hnsw.neighbors.size())}
            }); 
            commit_record.data_changes.emplace_back(
                append_target_addr,
                append_data_ptr,
                append_size_bytes
            );
        } else {
            // Overflow detected - trigger reconstruction
            std::cout << "[OVERFLOW] neighbors append overflow detected for sub_idx " << sub_idx << std::endl;
            overflow_detected_.store(true);
            last_overflow_type_ = "neighbors_append";
            last_overflow_sub_idx_ = sub_idx;
            if (overflow_callback_) {
                overflow_callback_(sub_idx, "neighbors_append");
            }
            return;  // Cannot proceed without reconstruction
        }
        }
    
        if(flat_storage->xb.size() > xb_before){
        size_t total_new_data_bytes = (flat_storage->xb.size() - xb_before) * sizeof(float);
        const float* new_data_ptr = reinterpret_cast<const float*>(
        flat_storage->xb.data() + xb_before
        ); 

        uint64_t internal_gap_end_addr = offset_para_[sub_idx * 9 + 8]; 
        uint64_t current_xb_end_addr =  offset_para_[sub_idx * 9 + 7] + xb_before * sizeof(float) + sizeof(uint64_t); 

        uint64_t available_internal_gap_bytes = internal_gap_end_addr - current_xb_end_addr;

        if (total_new_data_bytes <= available_internal_gap_bytes) {
            commit_record.overwrites.push_back(OverwriteChange{
                offset_para_[sub_idx * 9 + 7],
                std::variant<faiss::Index::idx_t, faiss::storage_idx_t, size_t>{static_cast<size_t>(flat_storage->xb.size())}
            }); 
            commit_record.data_changes.emplace_back(
                current_xb_end_addr,
                new_data_ptr,
                total_new_data_bytes
            );

        } else {
            // Overflow detected - trigger reconstruction
            std::cout << "[OVERFLOW] xb (vector data) overflow detected for sub_idx " << sub_idx << std::endl;
            overflow_detected_.store(true);
            last_overflow_type_ = "xb";
            last_overflow_sub_idx_ = sub_idx;
            if (overflow_callback_) {
                overflow_callback_(sub_idx, "xb");
            }
            return;  // Cannot proceed without reconstruction
        }
    } 

    // std::cout << "commit_record.overwrites.size() " << commit_record.overwrites.size() << std::endl;
    // std::cout << "commit_record.data_changes.size() " << commit_record.data_changes.size() << std::endl;
    // std::cout << "sub_idx " << sub_idx << std::endl;

    // for (size_t i = 0; i < commit_record.overwrites.size(); ++i) {
    //     const auto& change = commit_record.overwrites[i];
    //     std::cout << "i " << i << std::endl;
    //     std::cout << "commit_record.overwrites[i].server_target_offset " << change.server_target_offset << std::endl;
    //     std::cout << "commit_record.overwrites[i].value " << std::visit([](auto&& arg) -> size_t { 
    //         return static_cast<size_t>(arg); 
    //     }, change.value) << std::endl;
    // }
    // for (size_t i = 0; i < commit_record.data_changes.size(); ++i) {
    //     const auto& change = commit_record.data_changes[i];
    //     std::cout << "i " << i << std::endl;
    //     std::cout << "commit_record.data_changes[i].server_target_offset " << change.server_target_offset << std::endl;
    //     std::cout << "commit_record.data_changes[i].data_size_bytes " << change.data_size_bytes << std::endl;
    //     std::cout << "commit_record.data_changes[i].local_data_ptr " << change.local_data_ptr << std::endl;
    // }
    size_t total_required_size = 0;
    for (const auto& change : commit_record.overwrites) {
        // Get actual size of the contained value based on its type
        total_required_size += std::visit([](auto&& arg) -> size_t { 
            return sizeof(arg); 
        }, change.value);
    }
    for (const auto& change : commit_record.data_changes) {
        total_required_size += change.data_size_bytes;
    }

    if (total_required_size > local_mem_->sz) {
        std::cerr << "FATAL ERROR: RDMA buffer overflow detected in sub_idx " << sub_idx << "!" << std::endl;
        std::cerr << "  Required size: " << total_required_size << " bytes." << std::endl;
        std::cerr << "  Available size: " << local_mem_->sz << " bytes." << std::endl;

        abort(); 
    }
    
    std::vector<BenchOp<>> ops;
        ops.reserve(commit_record.overwrites.size() + commit_record.data_changes.size());
        uint8_t* local_rdma_buf = reinterpret_cast<uint8_t*>(local_mem_->raw_ptr);
        size_t local_buf_offset = 0;
        u32 lkey = local_mr->get_reg_attr().value().key;
        u32 rkey = remote_attr.key;
        uint64_t remote_base = reinterpret_cast<uint64_t>(remote_attr.buf);
        
        for (const auto& change : commit_record.overwrites) {
            // Extract the actual value and write it correctly
            std::visit([&](auto&& arg) {
                using T = std::decay_t<decltype(arg)>;
                memcpy(local_rdma_buf + local_buf_offset, &arg, sizeof(T));
                ops.emplace_back();
                ops.back().set_type(1); // WRITE
                ops.back().init_lbuf(reinterpret_cast<u64*>(local_rdma_buf + local_buf_offset), sizeof(T), lkey);
                ops.back().init_rbuf(reinterpret_cast<u64*>(remote_base + change.server_target_offset), rkey);
                local_buf_offset += sizeof(T);
            }, change.value);
        }

        for (const auto& change : commit_record.data_changes) {
            memcpy(local_rdma_buf + local_buf_offset, change.local_data_ptr, change.data_size_bytes);
            ops.emplace_back();
            ops.back().set_type(1); // WRITE
            ops.back().init_lbuf(reinterpret_cast<u64*>(local_rdma_buf + local_buf_offset), change.data_size_bytes, lkey);
            ops.back().init_rbuf(reinterpret_cast<u64*>(remote_base + change.server_target_offset), rkey);
            local_buf_offset += change.data_size_bytes;
        }

        constexpr size_t kMaxBatchSize = 199;

        if (!ops.empty()) {
            for (size_t start = 0; start < ops.size(); start += kMaxBatchSize) {
                // Check if overflow was detected (server may have crashed/restarted)
                if (overflow_detected_.load()) {
                    std::cerr << "[RDMA] Overflow detected, aborting RDMA writes for sub_idx "
                              << sub_idx << " (batch starting at " << start << ")" << std::endl;
                    return;
                }

                // compute [start, end) for this batch
                size_t end = std::min(start + kMaxBatchSize, ops.size());

                // chain ops within this batch
                for (size_t i = start; i + 1 < end; ++i) {
                    ops[i].set_flags(0);
                    ops[i].set_next(&ops[i + 1]);
                }

                // last op in this batch: signaled & terminate chain
                size_t last = end - 1;
                ops[last].set_flags(IBV_SEND_SIGNALED);
                ops[last].set_next(nullptr);  // important: do NOT chain into next batch

                try {
                    // execute this batch starting from ops[start]
                    auto res_s = ops[start].execute_batch(qp_shared);
                    if (res_s != IOCode::Ok) {
                        std::cerr << "[RDMA ERROR] Failed to execute RDMA write batch for sub_idx "
                                  << sub_idx << " (batch starting at " << start << "): "
                               << std::endl;
                        std::cerr << "[RDMA ERROR] Server may have crashed or connection lost. "
                                  << "Setting overflow flag to trigger reconstruction/reconnection." << std::endl;
                        overflow_detected_.store(true);
                        last_overflow_type_ = "rdma_execute_failure";
                        last_overflow_sub_idx_ = sub_idx;
                        if (overflow_callback_) {
                            overflow_callback_(sub_idx, "rdma_execute_failure");
                        }
                        return;  // Fail gracefully instead of asserting
                    }

                    auto res_p = qp_shared->wait_one_comp();
                    if (res_p != IOCode::Ok) {
                        std::cerr << "[RDMA ERROR] Failed to wait for RDMA write completion for sub_idx "
                                  << sub_idx << " (batch starting at " << start << "): "
                                  << std::endl;
                        std::cerr << "[RDMA ERROR] Server may have crashed or QP entered error state. "
                                  << "Setting overflow flag to trigger reconstruction/reconnection." << std::endl;
                        overflow_detected_.store(true);
                        last_overflow_type_ = "rdma_completion_failure";
                        last_overflow_sub_idx_ = sub_idx;
                        if (overflow_callback_) {
                            overflow_callback_(sub_idx, "rdma_completion_failure");
                        }
                        return;  // Fail gracefully instead of asserting
                    }
                } catch (const std::runtime_error &e) {
                    std::cerr << "[RDMA EXCEPTION] RDMA operation failed for sub_idx " << sub_idx
                              << " in batch starting at " << start << ": "
                              << e.what() << std::endl;
                    std::cerr << "[RDMA EXCEPTION] Setting overflow flag to trigger recovery." << std::endl;
                    overflow_detected_.store(true);
                    last_overflow_type_ = "rdma_exception";
                    last_overflow_sub_idx_ = sub_idx;
                    if (overflow_callback_) {
                        overflow_callback_(sub_idx, "rdma_exception");
                    }
                    return;  // Fail gracefully
                }
            }
        }
        // std::cout << "sub_idx " << sub_idx << " insert success" << std::endl;

}
