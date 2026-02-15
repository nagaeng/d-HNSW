// client_insert_reconstruction.cc
// Master computing node with insert operations and reconstruction triggering

#include <gflags/gflags.h>
#include <grpcpp/grpcpp.h>
#include "../generated/dhnsw.grpc.pb.h"
#include "google/protobuf/empty.pb.h"

#include "../../deps/rlib/core/lib.hh"
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <chrono>
#include <pthread.h>
#include <numeric>
#include <fstream>
#include <atomic>
#include <random>
#include <unordered_set>
#include <omp.h>
#include "../dhnsw/statics.hh"
#include "../dhnsw/reporter.hh"
#include "../dhnsw/DistributedHnsw.h"
#include "../dhnsw/reconstruction.hh"
#include "../util/read_dataset.h"

typedef int64_t dhnsw_idx_t;

DEFINE_string(server_address, "0.0.0.0:50051", "Address of the gRPC server.");
DEFINE_string(rdma_server_address, "192.168.1.2:8888", "Address of the RDMA server (InfiniBand IP).");
DEFINE_int32(use_nic_idx, 3, "Which NIC to create QP");
DEFINE_int32(reg_nic_name, 73, "The name to register an opened NIC at rctrl.");
DEFINE_int32(reg_mem_name, 73, "The name to register an MR at rctrl.");
DEFINE_string(query_data_path, "../datasets/sift/sift_query.fvecs", "Path to the query data.");
DEFINE_string(ground_truth_path, "../datasets/sift/sift_groundtruth.ivecs", "Path to the ground truth data.");
DEFINE_int32(benchmark_duration, 60, "Duration (in seconds) to run benchmark.");
DEFINE_int32(physical_cores_per_thread, 36, "Number of physical cores per thread");
DEFINE_bool(use_physical_cores_only, true, "Whether to use only physical cores");
DEFINE_string(log_file, "master_insert.log", "Path to the log file.");
DEFINE_string(throughput_log, "reconstruction/master_throughput.csv", "Path to throughput log.");
DEFINE_int32(insert_percentage, 10, "Percentage of insert operations (0-100)");
DEFINE_int32(batch_size, 5000, "Batch size for operations");
DEFINE_double(overflow_threshold, 1, "Overflow threshold for triggering reconstruction");
DEFINE_string(client_id, "master_insert", "Client identifier");
DEFINE_int32(dim, 128, "Vector dimension");
DEFINE_int32(num_sub_hnsw, 160, "Number of sub-HNSW indices");

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using dhnsw::DhnswService;
using dhnsw::Empty;
using dhnsw::MetaHnswResponse;
using dhnsw::Offset_ParaResponse;
using dhnsw::Offset_SubHnswResponse;
using dhnsw::MappingResponse;
using dhnsw::MappingEntry;
using dhnsw::OverflowResponse;

using namespace rdmaio;
using namespace rdmaio::rmem;
using namespace rdmaio::qp;
using namespace std;
using namespace std::chrono;
using namespace dhnsw::reconstruction;

// Global dataset variables
std::vector<float> query_data;
std::vector<int> ground_truth;
int dim_query_data;
int n_query_data;
int dim_ground_truth;

// Thread parameters
struct thread_param_t {
    int thread_id;
    int omp_threads_per_worker;
    int core_start;
    double latency;
    int query_start;
    int query_end;
    std::vector<float> per_ef_recalls;
    std::vector<double> per_ef_latencies;
    std::vector<double> per_ef_network_latencies;
    std::vector<double> per_ef_duration_meta_search;
    std::vector<double> per_ef_compute_times;
    std::vector<double> per_ef_deserialize_times;
    std::vector<double> per_ef_throughput;
    Arc<RMem> local_mem;
    Arc<RegHandler> local_mr;
    Arc<RNic> nic;
    std::shared_ptr<RC> qp;
    rmem::RegAttr remote_attr;
    uint64_t key;
};

// Reconstruction state for master node
struct MasterReconstructionState {
    std::atomic<bool> reconstruction_in_progress{false};
    std::atomic<bool> need_reconstruction{false};

    // Epoch tracking
    std::atomic<uint64_t> current_epoch{0};

    // LSH cache for buffering inserts during reconstruction
    std::unique_ptr<LSHCache> insert_cache;
    std::mutex cache_mutex;

    // Throughput logger
    std::unique_ptr<ThroughputLogger> throughput_logger;

    // Client reconstruction handler
    std::unique_ptr<ClientReconstructionHandler> handler;

    // Current overflow status
    std::vector<uint64_t> current_overflow;
    std::mutex overflow_mutex;

    // Statistics
    std::atomic<int64_t> total_inserts{0};
    std::atomic<int64_t> total_queries{0};
    std::atomic<int64_t> blocked_duration_us{0};
};

MasterReconstructionState g_reconstruction_state;

void* client_worker(void* param);

void bind_thread_to_cores(int thread_id, int core_start, int cores_per_thread, bool physical_cores_only) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    if (physical_cores_only) {
        for (int i = 0; i < cores_per_thread; i++) {
            int physical_core = core_start + (i * 2);
            CPU_SET(physical_core, &cpuset);
        }
    } else {
        for (int i = 0; i < cores_per_thread; i++) {
            CPU_SET(core_start + i, &cpuset);
        }
    }
    
    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error: unable to set thread affinity for thread " << thread_id 
                  << ", error code: " << rc << std::endl;
    }
}

// Check if reconstruction is needed based on overflow detection
bool check_reconstruction_needed(LocalHnsw* local_hnsw) {
    // Check if overflow was detected during insert operations
    return local_hnsw->has_overflow_detected();
}

// Overflow callback handler - called when gap is full
void handle_overflow_detected(int sub_idx, const std::string& overflow_type) {
    std::cout << "[Master] Overflow detected in sub_idx " << sub_idx 
              << " (type: " << overflow_type << ")" << std::endl;
    g_reconstruction_state.need_reconstruction.store(true);
}

// Structured log helper for reconstruction state machine
void log_reconstruction_state(const std::string& event,
                              bool reconstruction_in_progress,
                              bool need_reconstruction,
                              uint64_t reconstruction_id,
                              int cache_size,
                              int64_t total_inserts) {
    auto now_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    std::cout << "{\"event\":\"" << event << "\""
              << ",\"timestamp_ms\":" << now_ms
              << ",\"reconstruction_in_progress\":" << (reconstruction_in_progress ? "true" : "false")
              << ",\"need_reconstruction\":" << (need_reconstruction ? "true" : "false")
              << ",\"reconstruction_id\":" << reconstruction_id
              << ",\"cache_size\":" << cache_size
              << ",\"total_inserts\":" << total_inserts
              << ",\"client_id\":\"" << FLAGS_client_id << "\"}" << std::endl;
}

// Current reconstruction ID (for tracking version)
static std::atomic<uint64_t> g_current_reconstruction_id{0};

// Trigger reconstruction on the server
bool trigger_server_reconstruction(DhnswClient* client) {
    auto start_time = high_resolution_clock::now();

    // Get buffered vectors from LSH cache
    std::vector<float> buffered_vectors;
    std::vector<int64_t> buffered_ids;
    int cache_size = 0;
    {
        std::lock_guard<std::mutex> lock(g_reconstruction_state.cache_mutex);
        if (g_reconstruction_state.insert_cache) {
            buffered_vectors = g_reconstruction_state.insert_cache->get_all_vectors();
            buffered_ids = g_reconstruction_state.insert_cache->get_all_ids();
            cache_size = g_reconstruction_state.insert_cache->size();
        }
    }

    int vector_count = buffered_vectors.size() / FLAGS_dim;

    // Log reconstruction trigger with structured format
    log_reconstruction_state("RECONSTRUCTION_TRIGGERED",
                            true, true, 0, cache_size,
                            g_reconstruction_state.total_inserts.load());

    g_reconstruction_state.reconstruction_in_progress.store(true);

    // Block queries during transition
    if (g_reconstruction_state.handler) {
        g_reconstruction_state.handler->enter_blocking_mode();
    }
    g_reconstruction_state.throughput_logger->set_blocked(true);
    g_reconstruction_state.throughput_logger->record_event("RECONSTRUCTION_START");

    // Call actual TriggerReconstruction RPC
    auto rpc_result = client->TriggerReconstruction(buffered_vectors, vector_count, FLAGS_client_id);

    if (rpc_result.success) {
        g_current_reconstruction_id.store(rpc_result.reconstruction_id);

        log_reconstruction_state("RECONSTRUCTION_RPC_SUCCESS",
                                true, false, rpc_result.reconstruction_id, cache_size,
                                g_reconstruction_state.total_inserts.load());

        // Poll for completion using epoch-based polling
        // Use much longer timeout - reconstruction can take 60+ seconds for large indices
        int poll_count = 0;
        const int max_polls = 1200;  // 120 seconds max wait (2 minutes)
        uint64_t new_epoch = 0;
        uint64_t current_epoch = g_reconstruction_state.current_epoch.load();
        
        std::cout << "[Master] Waiting for reconstruction to complete (current epoch: " 
                  << current_epoch << ")..." << std::endl;
        
        while (poll_count < max_polls) {
            // Get epoch info
            auto epoch_info = client->GetEpochInfo();
            
            // Check if reconstruction completed and epoch advanced
            if (!epoch_info.reconstruction_in_progress &&
                epoch_info.epoch > current_epoch) {
                new_epoch = epoch_info.epoch;
                std::cout << "[Master] Reconstruction complete, new epoch: " << new_epoch << std::endl;
                log_reconstruction_state("RECONSTRUCTION_COMPLETED",
                                        false, false, rpc_result.reconstruction_id, 0,
                                        g_reconstruction_state.total_inserts.load());
                break;
            }
            
            // Also check reconstruction status for progress feedback
            if (poll_count % 10 == 0) {  // Every 1 second
                auto status = client->GetReconstructionStatus();
                if (poll_count % 50 == 0) {  // Every 5 seconds
                    std::cout << "[Master] Reconstruction progress: " << status.progress 
                              << "% - " << status.message << std::endl;
                }
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            poll_count++;
        }

        // If we timed out, still wait a bit more for the epoch to change
        if (new_epoch == 0) {
            std::cout << "[Master] Polling timeout reached, doing final epoch check..." << std::endl;
            for (int retry = 0; retry < 50; retry++) {  // 5 more seconds
                auto epoch_info = client->GetEpochInfo();
                if (epoch_info.epoch > current_epoch) {
                    new_epoch = epoch_info.epoch;
                    std::cout << "[Master] Found new epoch after timeout: " << new_epoch << std::endl;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }

        // Update local epoch
        if (new_epoch > 0) {
            g_reconstruction_state.current_epoch.store(new_epoch);
            // Send acknowledgment to server with NEW epoch
            bool all_acked = client->AcknowledgeReconstruction(FLAGS_client_id, rpc_result.reconstruction_id, new_epoch);
            std::cout << "[Master] Sent acknowledgment for epoch " << new_epoch 
                      << ", all_acknowledged: " << (all_acked ? "true" : "false") << std::endl;
        } else {
            std::cerr << "[Master] ERROR: Reconstruction may have failed - epoch didn't advance!" << std::endl;
            std::cerr << "[Master] Will retry with current data..." << std::endl;
            // Don't send ack for epoch 0 as that confuses the server
        }

    } else {
        // RPC failed - check if it's "already in progress" which we should wait for
        log_reconstruction_state("RECONSTRUCTION_RPC_FAILED",
                                true, true, 0, cache_size,
                                g_reconstruction_state.total_inserts.load());

        std::cerr << "[Master] WARNING: TriggerReconstruction RPC failed: " << rpc_result.message << std::endl;
        
        if (rpc_result.message.find("already in progress") != std::string::npos) {
            // Another reconstruction is in progress - wait for it to complete
            std::cout << "[Master] Reconstruction already in progress, waiting for completion..." << std::endl;
            
            uint64_t current_epoch = g_reconstruction_state.current_epoch.load();
            for (int wait = 0; wait < 600; wait++) {  // Wait up to 60 seconds
                auto epoch_info = client->GetEpochInfo();
                if (!epoch_info.reconstruction_in_progress && epoch_info.epoch > current_epoch) {
                    g_reconstruction_state.current_epoch.store(epoch_info.epoch);
                    std::cout << "[Master] Existing reconstruction completed, new epoch: " 
                              << epoch_info.epoch << std::endl;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        } else {
            std::cerr << "[Master] Falling back to simulated reconstruction..." << std::endl;
            // Fallback: simulate with delay (for when server doesn't have handler)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }

    auto end_time = high_resolution_clock::now();
    auto duration_ms = duration_cast<milliseconds>(end_time - start_time).count();

    // Clear reconstruction state after completion
    {
        std::lock_guard<std::mutex> lock(g_reconstruction_state.cache_mutex);
        if (g_reconstruction_state.insert_cache) {
            g_reconstruction_state.insert_cache->clear();
        }
    }

    g_reconstruction_state.reconstruction_in_progress.store(false);
    g_reconstruction_state.need_reconstruction.store(false);

    // Exit blocking mode
    if (g_reconstruction_state.handler) {
        g_reconstruction_state.handler->exit_blocking_mode();
    }
    g_reconstruction_state.throughput_logger->set_blocked(false);
    g_reconstruction_state.throughput_logger->record_event("RECONSTRUCTION_END");

    log_reconstruction_state("RECONSTRUCTION_COMPLETE",
                            false, false, g_current_reconstruction_id.load(), 0,
                            g_reconstruction_state.total_inserts.load());

    std::cout << "[Master] Reconstruction completed in " << duration_ms << " ms" << std::endl;

    return rpc_result.success;
}

// Handle reconstruction completion
void handle_reconstruction_complete(const ReconstructionState& state) {
    std::cout << "[Master] Reconstruction complete, updating local state..." << std::endl;
    
    // Clear the insert cache
    {
        std::lock_guard<std::mutex> lock(g_reconstruction_state.cache_mutex);
        if (g_reconstruction_state.insert_cache) {
            g_reconstruction_state.insert_cache->clear();
        }
    }
    
    g_reconstruction_state.reconstruction_in_progress.store(false);
    g_reconstruction_state.need_reconstruction.store(false);
    
    // Exit blocking mode
    if (g_reconstruction_state.handler) {
        g_reconstruction_state.handler->exit_blocking_mode();
    }
    
    std::cout << "[Master] Local state updated" << std::endl;
}

// Add vectors to insert cache during reconstruction
void buffer_insert_during_reconstruction(const float* vectors, int count, int dim) {
    std::lock_guard<std::mutex> lock(g_reconstruction_state.cache_mutex);
    
    if (!g_reconstruction_state.insert_cache) {
        g_reconstruction_state.insert_cache = std::make_unique<LSHCache>(dim);
    }
    
    std::vector<int64_t> ids(count);
    std::iota(ids.begin(), ids.end(), g_reconstruction_state.total_inserts.load());
    
    g_reconstruction_state.insert_cache->add_batch(ids, vectors, count);
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Initialize reconstruction state
    g_reconstruction_state.insert_cache = std::make_unique<LSHCache>(FLAGS_dim);
    g_reconstruction_state.handler = std::make_unique<ClientReconstructionHandler>(
        FLAGS_client_id, true /* is_master */);
    g_reconstruction_state.handler->set_update_callback(handle_reconstruction_complete);
    g_reconstruction_state.throughput_logger = std::make_unique<ThroughputLogger>(
        FLAGS_throughput_log, FLAGS_client_id);
    g_reconstruction_state.throughput_logger->start_logging(100);  // 100ms interval

    // Print configuration
    int query_percentage = 100 - FLAGS_insert_percentage;
    std::cout << "=== Master Insert Node Configuration ===" << std::endl;
    std::cout << "Client ID: " << FLAGS_client_id << std::endl;
    std::cout << "Insert percentage: " << FLAGS_insert_percentage << "%" << std::endl;
    std::cout << "Query percentage: " << query_percentage << "%" << std::endl;
    std::cout << "Batch size: " << FLAGS_batch_size << std::endl;
    std::cout << "Overflow threshold: " << FLAGS_overflow_threshold << std::endl;
    std::cout << "==========================================" << std::endl << std::endl;

    // Read dataset
    std::string query_data_path = FLAGS_query_data_path;
    std::string ground_truth_path = FLAGS_ground_truth_path;
    std::vector<float> query_data_tmp = read_fvecs(query_data_path, dim_query_data, n_query_data);
    std::vector<int> ground_truth_tmp = read_ivecs(ground_truth_path, dim_ground_truth, n_query_data);
    
    // Sample query data
    n_query_data = 0;
    for (int i = 0; i < (int)query_data_tmp.size() / dim_query_data; i++) {
        if (i % 3 == 0) {
            query_data.insert(query_data.end(), 
                              query_data_tmp.begin() + i * dim_query_data,
                              query_data_tmp.begin() + (i + 1) * dim_query_data);
            ground_truth.insert(ground_truth.end(),
                                ground_truth_tmp.begin() + i * dim_ground_truth,
                                ground_truth_tmp.begin() + (i + 1) * dim_ground_truth);
            n_query_data++;
        }
    }
    
    // Replicate for longer benchmark
    int original_n_query_data = n_query_data;
    std::vector<float> original_query_data = query_data;
    std::vector<int> original_ground_truth = ground_truth;
    for (int rep = 1; rep < 1000; rep++) {
        query_data.insert(query_data.end(), original_query_data.begin(), original_query_data.end());
        ground_truth.insert(ground_truth.end(), original_ground_truth.begin(), original_ground_truth.end());
    }
    n_query_data = original_n_query_data * 1000;
    
    int num_threads = 1;
    int queries_per_thread = n_query_data / num_threads;
    ground_truth.resize(n_query_data * dim_ground_truth);
    
    int omp_threads_per_worker = FLAGS_physical_cores_per_thread;
    
    std::vector<pthread_t> threads(num_threads);
    std::vector<thread_param_t> thread_params(num_threads);
    
    // Initialize RDMA resources
    ConnectManager cm(FLAGS_rdma_server_address);
    if (cm.wait_ready(1000000, 2) == IOCode::Timeout) {
        RDMA_LOG(4) << "cm connect to server timeout";
        return -1;
    }
    
    auto fetch_res = cm.fetch_remote_mr(FLAGS_reg_mem_name);
    RDMA_ASSERT(fetch_res == IOCode::Ok) << std::get<0>(fetch_res.desc);
    rmem::RegAttr remote_attr = std::get<1>(fetch_res.desc);
    
    std::vector<Arc<RNic>> nics(num_threads);
    std::vector<std::shared_ptr<RC>> qps(num_threads);
    std::vector<Arc<RMem>> local_mems(num_threads);
    std::vector<Arc<RegHandler>> local_mrs(num_threads);
    std::vector<uint64_t> keys(num_threads);
    
    auto timestamp = duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
    
    for (int i = 0; i < num_threads; ++i) {
        nics[i] = RNic::create(RNicInfo::query_dev_names().at(FLAGS_use_nic_idx)).value();
        qps[i] = RC::create(nics[i], QPConfig()).value();
        std::string qp_name = "-master-insert@" + std::to_string(timestamp) + std::to_string(i);
        auto qp_res = cm.cc_rc(qp_name, qps[i], FLAGS_reg_nic_name, QPConfig());
        RDMA_ASSERT(qp_res == IOCode::Ok) << std::get<0>(qp_res.desc);
        keys[i] = std::get<1>(qp_res.desc);
        
        size_t fixed_size = 2UL * 1024 * 1024 * 1024;
        local_mems[i] = Arc<RMem>(new RMem(fixed_size));
        local_mrs[i] = RegHandler::create(local_mems[i], nics[i]).value();
    }
    
    // Create worker threads
    for (int i = 0; i < num_threads; ++i) {
        thread_params[i].thread_id = i;
        thread_params[i].latency = 0;
        thread_params[i].omp_threads_per_worker = omp_threads_per_worker;
        
        if (FLAGS_use_physical_cores_only) {
            thread_params[i].core_start = i * omp_threads_per_worker * 2;
        } else {
            thread_params[i].core_start = i * omp_threads_per_worker;
        }
        
        thread_params[i].query_start = i * queries_per_thread;
        thread_params[i].query_end = thread_params[i].query_start + queries_per_thread;
        thread_params[i].nic = nics[i];
        thread_params[i].qp = qps[i];
        thread_params[i].local_mem = local_mems[i];
        thread_params[i].local_mr = local_mrs[i];
        thread_params[i].remote_attr = remote_attr;
        thread_params[i].key = keys[i];
        
        int ret = pthread_create(&threads[i], nullptr, client_worker, (void*)&thread_params[i]);
        if (ret != 0) {
            std::cerr << "Error: unable to create thread, " << ret << std::endl;
            exit(-1);
        }
    }
    
    // Wait for threads
    for (int i = 0; i < num_threads; ++i) {
        void* status;
        pthread_join(threads[i], &status);
    }
    
    // Stop throughput logging
    g_reconstruction_state.throughput_logger->stop_logging();
    
    // Output final statistics
    std::cout << "\n=== Final Statistics ===" << std::endl;
    std::cout << "Total inserts: " << g_reconstruction_state.total_inserts.load() << std::endl;
    std::cout << "Total queries: " << g_reconstruction_state.total_queries.load() << std::endl;
    std::cout << "Total blocked time: " << g_reconstruction_state.blocked_duration_us.load() << " us" << std::endl;
    
    // Write results
    std::vector<int> ef_search_values = {48, 48, 48};
    std::vector<float> avg_recalls;
    std::vector<double> avg_throughput;
    
    for (size_t ef_idx = 0; ef_idx < ef_search_values.size(); ++ef_idx) {
        float sum_recalls = 0.0f;
        double sum_throughput = 0.0;
        
        for (int t = 0; t < num_threads; ++t) {
            if (ef_idx < thread_params[t].per_ef_recalls.size()) {
                sum_recalls += thread_params[t].per_ef_recalls[ef_idx];
                sum_throughput += thread_params[t].per_ef_throughput[ef_idx];
            }
        }
        avg_recalls.push_back(sum_recalls / num_threads);
        avg_throughput.push_back(sum_throughput / num_threads);
    }

    std::ofstream outfile("../benchs/reconstruction/master_results.txt");
    outfile << "# Master Insert Node Results" << std::endl;
    outfile << "# Insert: " << FLAGS_insert_percentage << "%, Query: " 
            << (100 - FLAGS_insert_percentage) << "%" << std::endl;
    outfile << "throughput(ops/s)\trecall" << std::endl;
    for (size_t i = 0; i < ef_search_values.size(); ++i) {
        outfile << "[" << avg_throughput[i] << ", " << avg_recalls[i] << "]," << std::endl;
    }
    outfile.close();

    // Cleanup
    for (int i = 0; i < num_threads; ++i) {
        std::string qp_name = "-master-insert@" + std::to_string(timestamp) + std::to_string(i);
        auto del_res = cm.delete_remote_rc(qp_name, thread_params[i].key);
        qps[i].reset();
    }
    
    return 0;
}

void* client_worker(void* param) {
    thread_param_t& thread_param = *(thread_param_t*)param;
    int thread_id = thread_param.thread_id;
    
    bind_thread_to_cores(thread_id, thread_param.core_start, 
                        thread_param.omp_threads_per_worker, FLAGS_use_physical_cores_only);
    omp_set_num_threads(thread_param.omp_threads_per_worker);
    
    std::vector<int> ef_search_values = {48, 48, 48};
    
    // Initialize LocalHnsw
    int dim = FLAGS_dim;
    int num_sub_hnsw = FLAGS_num_sub_hnsw;
    int meta_hnsw_neighbors = 16;
    int sub_hnsw_neighbors = 32;
    
    DhnswClient* dhnsw_client = new DhnswClient(
        grpc::CreateChannel(FLAGS_server_address, grpc::InsecureChannelCredentials()));
    LocalHnsw local_hnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, dhnsw_client);
    local_hnsw.init();
    
    // Set overflow callback to trigger reconstruction when gap is full
    local_hnsw.set_overflow_callback(handle_overflow_detected);
    
    std::cout << "Thread " << thread_id << " initialized, mapping size: " 
              << local_hnsw.get_local_mapping().size() << std::endl;
    
    local_hnsw.set_rdma_qp(thread_param.qp, thread_param.remote_attr, thread_param.local_mr);
    local_hnsw.set_remote_attr(thread_param.remote_attr);
    local_hnsw.set_local_mr(thread_param.local_mr, thread_param.local_mem);

    PipelinedSearchManager search_manager(&local_hnsw, thread_param.core_start, 
                                         thread_param.omp_threads_per_worker);
    
    int query_start = thread_param.query_start;
    int query_end = thread_param.query_end;
    int n_query_data_thread = query_end - query_start;
    const float* query_data_ptr = query_data.data() + query_start * dim_query_data;
    const int* ground_truth_ptr = ground_truth.data() + query_start * dim_ground_truth;
    
    auto run_ef_benchmark = [&](int ef, size_t duration_sec) {
        int batch_size = FLAGS_batch_size;
        int top_k = 1;
        int branching_k = 5;
        int queries_executed = 0;
        int inserts_executed = 0;
        double total_ops_time = 0.0;
        double total_compute_time = 0.0;
        double total_network_latency = 0.0;
        double total_meta_search_time = 0.0;
        double total_deserialize_time = 0.0;
        
        std::vector<int> all_retrieved;
        std::vector<int> all_ground_truth;
        
        float* batch_meta_distances = new float[branching_k * batch_size];
        dhnsw_idx_t* batch_meta_labels = new dhnsw_idx_t[branching_k * batch_size];
        dhnsw_idx_t* batch_sub_hnsw_tags = new dhnsw_idx_t[top_k * batch_size];
        dhnsw_idx_t* batch_labels = new dhnsw_idx_t[top_k * batch_size];
        float* batch_distances = new float[top_k * batch_size];

        auto bench_start = high_resolution_clock::now();
        int query_index = 0;
        
        while (duration_cast<seconds>(high_resolution_clock::now() - bench_start).count() < (long)duration_sec) {
            // Check if reconstruction is in progress
            if (g_reconstruction_state.reconstruction_in_progress.load()) {
                auto block_start = high_resolution_clock::now();
                
                g_reconstruction_state.throughput_logger->set_blocked(true);
                
                // Wait for reconstruction to complete with timeout
                constexpr int MAX_BLOCK_MS = 5000;
                int waited_ms = 0;
                while (g_reconstruction_state.reconstruction_in_progress.load() && waited_ms < MAX_BLOCK_MS) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    waited_ms += 50;
                }
                
                if (waited_ms >= MAX_BLOCK_MS) {
                    std::cerr << "[Worker] Reconstruction wait timeout, continuing..." << std::endl;
                    g_reconstruction_state.reconstruction_in_progress.store(false);
                }
                
                g_reconstruction_state.throughput_logger->set_blocked(false);
                
                auto block_end = high_resolution_clock::now();
                auto blocked_us = duration_cast<microseconds>(block_end - block_start).count();
                g_reconstruction_state.blocked_duration_us.fetch_add(blocked_us);
                
                // Safe re-initialization: wait for any active RDMA reads to complete
                local_hnsw.safe_reinit();
                std::cout << "[Worker] Safely reinitialized after reconstruction" << std::endl;
            }
            
            // Check if overflow was detected (gap is full) - need to trigger reconstruction
            if ((check_reconstruction_needed(&local_hnsw) || 
                 g_reconstruction_state.need_reconstruction.load()) && 
                !g_reconstruction_state.reconstruction_in_progress.load()) {
                
                std::cout << "[Master] Overflow detected, triggering reconstruction..." << std::endl;
                std::cout << "  Overflow type: " << local_hnsw.get_last_overflow_type() << std::endl;
                std::cout << "  Sub-index: " << local_hnsw.get_last_overflow_sub_idx() << std::endl;
                
                // Clear the overflow flag
                local_hnsw.clear_overflow_flag();
                g_reconstruction_state.need_reconstruction.store(false);

                bool reconstruction_success = trigger_server_reconstruction(dhnsw_client);

                // Reinitialize LocalHnsw with updated metadata from server
                // Wait for the new epoch to be available before reinitializing
                uint64_t expected_epoch = g_reconstruction_state.current_epoch.load();
                
                if (reconstruction_success) {
                    // Do a loop to ensure we get the new epoch
                    int reinit_attempts = 0;
                    const int max_reinit_attempts = 10;
                    
                    while (reinit_attempts < max_reinit_attempts) {
                        local_hnsw.init();
                        
                        // Check if we got the expected epoch
                        uint64_t got_epoch = local_hnsw.get_current_epoch();
                        if (got_epoch >= expected_epoch) {
                            std::cout << "[Master] Reinitialized LocalHnsw after reconstruction (epoch: " 
                                      << got_epoch << ")" << std::endl;
                            break;
                        }
                        
                        std::cout << "[Master] Got stale epoch " << got_epoch 
                                  << ", expected >= " << expected_epoch << ", retrying..." << std::endl;
                        std::this_thread::sleep_for(std::chrono::milliseconds(200));
                        reinit_attempts++;
                    }
                    
                    if (reinit_attempts >= max_reinit_attempts) {
                        std::cerr << "[Master] WARNING: Could not get expected epoch after " 
                                  << max_reinit_attempts << " attempts!" << std::endl;
                    }
                } else {
                    // Reconstruction failed - just reinit with current data
                    // The overflow flag might still be set, so we might trigger again
                    std::cerr << "[Master] Reconstruction failed, reinitializing with current data..." << std::endl;
                    local_hnsw.init();
                }
                continue;
            }
            
            int current_batch_size = std::min(batch_size, 
                n_query_data_thread - (query_index % n_query_data_thread));
            if (current_batch_size <= 0) {
                current_batch_size = batch_size;
            }
            
            const float* batch_query_data_ptr = query_data_ptr + 
                ((query_index % n_query_data_thread) * dim_query_data);
            
            int insert_count = (current_batch_size * FLAGS_insert_percentage) / 100;
            int query_count = current_batch_size - insert_count;
            
            // Record batch start time
            auto batch_start_time = high_resolution_clock::now();
            uint64_t batch_start_ms = duration_cast<milliseconds>(
                batch_start_time.time_since_epoch()).count();
            
            // INSERT OPERATIONS
            if (insert_count > 0) {
                if (g_reconstruction_state.reconstruction_in_progress.load()) {
                    // Buffer inserts during reconstruction
                    buffer_insert_during_reconstruction(batch_query_data_ptr, insert_count, dim);
                    inserts_executed += insert_count;
                } else {
                    const int max_insert_batch = 60;
                    int remaining = insert_count;
                    int offset = 0;
                    
                    while (remaining > 0) {
                        int current_insert_batch = std::min(max_insert_batch, remaining);
                        const float* insert_data_ptr = batch_query_data_ptr + offset * dim_query_data;
                        std::vector<float> batch_insert_data(insert_data_ptr,
                            insert_data_ptr + current_insert_batch * dim_query_data);

                        local_hnsw.insert_to_server(current_insert_batch, batch_insert_data);

                        // Check if overflow was detected during insert
                        if (local_hnsw.has_overflow_detected()) {
                            std::cerr << "[INSERT] Overflow detected during batch insert, buffering remaining "
                                      << remaining << " vectors (including current batch)" << std::endl;
                            // Buffer the current batch that failed, plus any remaining inserts
                            const float* remaining_data_ptr = batch_query_data_ptr + offset * dim_query_data;
                            buffer_insert_during_reconstruction(remaining_data_ptr, remaining, dim);
                            inserts_executed += remaining;
                            break;  // Stop processing further inserts
                        }

                        inserts_executed += current_insert_batch;
                        remaining -= current_insert_batch;
                        offset += current_insert_batch;
                    }
                }
                
                g_reconstruction_state.total_inserts.fetch_add(insert_count);
            }
            
            // QUERY OPERATIONS
            if (query_count > 0 && !g_reconstruction_state.reconstruction_in_progress.load()) {
                const float* query_portion_ptr = batch_query_data_ptr + insert_count * dim_query_data;
                
                try {
                    std::vector<int> sub_hnsw_tosearch_batch;
                    std::unordered_map<int, std::unordered_set<int>> searchset;
                    
                    auto meta_start = high_resolution_clock::now();
                    searchset = local_hnsw.meta_search_pipelined(query_count, query_portion_ptr, 
                        branching_k, batch_meta_distances, batch_meta_labels, sub_hnsw_tosearch_batch, 
                        thread_param.core_start, thread_param.omp_threads_per_worker);
                    auto meta_end = high_resolution_clock::now();
                    total_meta_search_time += duration_cast<microseconds>(meta_end - meta_start).count();
                    
                    std::fill(batch_sub_hnsw_tags, batch_sub_hnsw_tags + top_k * query_count, -1);
                    
                    std::tuple<double, double, double> batch_result = search_manager.process_batch(
                        query_count, query_portion_ptr, top_k,
                        batch_distances, batch_labels, searchset, batch_sub_hnsw_tags, ef);
                    
                    total_compute_time += std::get<0>(batch_result);
                    total_network_latency += std::get<1>(batch_result);
                    total_deserialize_time += std::get<2>(batch_result);
                    
                    for (int i = 0; i < query_count; ++i) {
                        all_retrieved.push_back(batch_labels[i * top_k]);
                        int gt_idx = ((query_index % n_query_data_thread) + insert_count + i) * dim_ground_truth;
                        all_ground_truth.push_back(ground_truth_ptr[gt_idx]);
                    }
                    
                    queries_executed += query_count;
                    g_reconstruction_state.total_queries.fetch_add(query_count);
                } catch (const std::exception& e) {
                    // Handle RDMA/deserialization errors during reconstruction
                    std::cerr << "[Master] Query error: " << e.what() 
                              << " - likely due to reconstruction, skipping batch..." << std::endl;
                    // Don't reinit here - let the reconstruction flow handle it
                }
            }
            
            // Record batch end time
            auto batch_end_time = high_resolution_clock::now();
            uint64_t batch_end_ms = duration_cast<milliseconds>(
                batch_end_time.time_since_epoch()).count();
            
            // Record batch to throughput logger
            g_reconstruction_state.throughput_logger->record_batch(
                thread_id, batch_start_ms, batch_end_ms,
                query_count, insert_count,
                g_reconstruction_state.reconstruction_in_progress.load());
            
            auto batch_end = high_resolution_clock::now();
            total_ops_time += duration_cast<microseconds>(batch_end - batch_start_time).count();
            query_index += current_batch_size;
        }
        
        delete[] batch_meta_distances;
        delete[] batch_meta_labels;
        delete[] batch_sub_hnsw_tags;
        delete[] batch_labels;
        delete[] batch_distances;
        
        // Calculate recall
        int total_correct = 0;
        for (size_t i = 0; i < all_retrieved.size(); i++) {
            if (all_retrieved[i] == all_ground_truth[i]) {
                total_correct++;
            }
        }
        float recall = (all_retrieved.size() > 0) ? 
            static_cast<float>(total_correct) / all_retrieved.size() : 0.0f;
        
        int total_ops = queries_executed + inserts_executed;
        double throughput = (total_ops > 0) ? total_ops / (total_ops_time * 1e-6) : 0.0;
        
        std::cout << "Thread " << thread_id << " EF " << ef << " benchmark:" << std::endl;
        std::cout << "  Total ops: " << total_ops << " (Inserts: " << inserts_executed 
                  << ", Queries: " << queries_executed << ")" << std::endl;
        std::cout << "  Throughput: " << throughput << " ops/s" << std::endl;
        std::cout << "  Recall: " << recall << std::endl;
        
        thread_param.per_ef_latencies.push_back(total_ops_time);
        thread_param.per_ef_recalls.push_back(recall);
        thread_param.per_ef_network_latencies.push_back(
            queries_executed > 0 ? total_network_latency / queries_executed : 0.0);
        thread_param.per_ef_compute_times.push_back(
            queries_executed > 0 ? total_compute_time / queries_executed : 0.0);
        thread_param.per_ef_duration_meta_search.push_back(
            queries_executed > 0 ? total_meta_search_time / queries_executed : 0.0);
        thread_param.per_ef_deserialize_times.push_back(
            queries_executed > 0 ? total_deserialize_time / queries_executed : 0.0);
        thread_param.per_ef_throughput.push_back(throughput);
    };
    
    for (int ef : ef_search_values) {
        std::cout << "=== Thread " << thread_id << " testing EF = " << ef 
                  << " for " << FLAGS_benchmark_duration << " seconds ===" << std::endl;
        run_ef_benchmark(ef, FLAGS_benchmark_duration);
    }

    delete dhnsw_client;
    return nullptr;
}

