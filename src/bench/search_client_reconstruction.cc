// search_client_reconstruction.cc
// Worker computing node with query operations and reconstruction handling

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
#include "../dhnsw/data_config.hh"
#include "../util/read_dataset.h"

typedef int64_t dhnsw_idx_t;

DEFINE_string(server_address, "0.0.0.0:50051", "Address of the gRPC server.");
DEFINE_string(rdma_server_address, "192.168.1.2:8888", "Address of the RDMA server (InfiniBand IP).");
DEFINE_int32(use_nic_idx, 3, "Which NIC to create QP");
DEFINE_int32(reg_nic_name, 73, "The name to register an opened NIC at rctrl.");
DEFINE_int32(reg_mem_name, 73, "The name to register an MR at rctrl.");
DEFINE_int32(benchmark_duration, 60, "Duration (in seconds) to run each ef benchmark.");
DEFINE_int32(physical_cores_per_thread, 8, "Number of physical cores per thread");
DEFINE_bool(use_physical_cores_only, true, "Whether to use only physical cores");
DEFINE_string(log_file, "worker_search.log", "Path to the log file.");
DEFINE_string(throughput_log, "worker_throughput.csv", "Path to throughput log.");
DEFINE_string(client_id, "worker_1", "Client identifier");
DEFINE_int32(status_check_interval_ms, 100, "Interval for checking reconstruction status");

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using dhnsw::DhnswService;
using dhnsw::Empty;
using dhnsw::MetaHnswResponse;
using dhnsw::MappingResponse;
using dhnsw::MappingEntry;
using dhnsw::Offset_SubHnswResponse;
using dhnsw::Offset_ParaResponse;
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

// Helper function for NUMA core mapping
int nth_phys_core_on_numa1(int n) {
    return n * 2 + 1;          
}

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

// Worker reconstruction state
struct WorkerReconstructionState {
    std::atomic<bool> reconstruction_in_progress{false};
    std::atomic<bool> need_update{false};
    std::atomic<uint64_t> current_reconstruction_id{0};

    // Epoch tracking
    std::atomic<uint64_t> current_epoch{0};
    std::atomic<uint64_t> rdma_offset{0};

    // Leader election for refresh - prevents stampede where all threads try to refresh at once
    std::atomic<bool> refresh_in_progress{false};
    std::atomic<int64_t> last_refresh_timestamp_ms{0};  // Track when last refresh happened

    // Worker coordination for ACK - only send ACK after ALL workers have called init()
    std::atomic<int> num_workers{0};           // Total number of worker threads
    std::atomic<int> workers_refreshed{0};      // Number of workers that have called init() for current epoch
    std::atomic<uint64_t> refresh_epoch{0};     // Epoch that workers are refreshing to
    std::atomic<bool> ack_sent{false};          // Whether ACK has been sent for current epoch
    std::mutex refresh_mutex;                   // Protect refresh coordination
    std::condition_variable refresh_cv;         // Signal when all workers refreshed

    // LSH cache synchronized from master
    std::unique_ptr<LSHCache> insert_cache;
    std::mutex cache_mutex;

    // Throughput logger
    std::unique_ptr<ThroughputLogger> throughput_logger;

    // Client reconstruction handler
    std::unique_ptr<ClientReconstructionHandler> handler;

    // Statistics
    std::atomic<int64_t> total_queries{0};
    std::atomic<int64_t> blocked_duration_us{0};

    // Updated index data (received after reconstruction)
    std::vector<uint8_t> new_meta_hnsw;
    std::vector<size_t> new_offset_subhnsw;
    std::vector<size_t> new_offset_para;
    std::vector<size_t> new_overflow;
    std::vector<std::vector<dhnsw_idx_t>> new_mapping;
    std::mutex update_mutex;
};

WorkerReconstructionState g_reconstruction_state;

void* client_worker(void* param);

void bind_thread_to_cores(int thread_id, int core_start, int cores_per_thread, bool physical_cores_only) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < cores_per_thread; i++) {
        CPU_SET(core_start + i*2, &cpuset);
    }
    
    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error: unable to set thread affinity for thread " << thread_id 
                  << ", error code: " << rc << std::endl;
    }
}

// Structured log helper for worker reconstruction state
void log_worker_state(const std::string& event,
                      bool reconstruction_in_progress,
                      bool need_update,
                      uint64_t reconstruction_id,
                      int cache_size,
                      int64_t total_queries) {
    auto now_ms = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    std::cout << "{\"event\":\"" << event << "\""
              << ",\"timestamp_ms\":" << now_ms
              << ",\"reconstruction_in_progress\":" << (reconstruction_in_progress ? "true" : "false")
              << ",\"need_update\":" << (need_update ? "true" : "false")
              << ",\"reconstruction_id\":" << reconstruction_id
              << ",\"cache_size\":" << cache_size
              << ",\"total_queries\":" << total_queries
              << ",\"client_id\":\"" << FLAGS_client_id << "\"}" << std::endl;
}

// Poll server for reconstruction status - epoch-based polling
void reconstruction_status_poller(DhnswClient* client) {
    log_worker_state("STATUS_POLLER_STARTED",
                    false, false, 0, 0,
                    g_reconstruction_state.total_queries.load());

    uint64_t last_epoch = 0;
    int consecutive_failures = 0;
    const int MAX_CONSECUTIVE_FAILURES = 10;

    while (true) {
        std::this_thread::sleep_for(std::chrono::milliseconds(FLAGS_status_check_interval_ms));

        // Step 1: Get epoch info first (with error handling)
        auto epoch_info = client->GetEpochInfo();

        // Handle RPC failures based on gRPC status, NOT returned values
        // Note: (epoch=0, rdma_offset=0, reconstruction_id=0) is a VALID initial state!
        if (!epoch_info.success) {
            consecutive_failures++;
            if (consecutive_failures >= MAX_CONSECUTIVE_FAILURES) {
                std::cerr << "[Poller] " << MAX_CONSECUTIVE_FAILURES
                          << " consecutive GetEpochInfo RPC failures, clearing reconstruction flag"
                          << std::endl;
                g_reconstruction_state.reconstruction_in_progress.store(false);
                consecutive_failures = 0;
            }
            continue;
        }
        consecutive_failures = 0;

        // Step 2: Check if epoch changed OR if reconstruction_in_progress cleared on server
        if (epoch_info.epoch > last_epoch) {
            // Epoch changed - new reconstruction completed
            // DON'T fetch metadata here - just signal workers to reinit
            // Workers will use AcquireEpochRead() to get all metadata atomically
            log_worker_state("EPOCH_CHANGED",
                            false, true, epoch_info.reconstruction_id, 0,
                            g_reconstruction_state.total_queries.load());

            // Setup coordination for worker refresh
            // Workers must all call init() BEFORE we send ACK to server
            {
                std::lock_guard<std::mutex> lock(g_reconstruction_state.refresh_mutex);
                g_reconstruction_state.refresh_epoch.store(epoch_info.epoch);
                g_reconstruction_state.workers_refreshed.store(0);
                g_reconstruction_state.ack_sent.store(false);
            }

            // Update epoch tracking and signal workers
            g_reconstruction_state.reconstruction_in_progress.store(false);
            g_reconstruction_state.current_epoch.store(epoch_info.epoch);
            g_reconstruction_state.rdma_offset.store(epoch_info.rdma_offset);
            g_reconstruction_state.current_reconstruction_id.store(epoch_info.reconstruction_id);
            g_reconstruction_state.need_update.store(true);

            // DO NOT send ACK here - wait for all workers to complete init()
            // Workers will signal when they're done, and the last one will send ACK
            std::cout << "[Poller] Epoch changed to " << epoch_info.epoch 
                      << ", waiting for " << g_reconstruction_state.num_workers.load()
                      << " workers to refresh before ACK..." << std::endl;

            log_worker_state("WAITING_FOR_WORKERS",
                            false, true, epoch_info.reconstruction_id, 0,
                            g_reconstruction_state.total_queries.load());

            last_epoch = epoch_info.epoch;

        } else if (epoch_info.reconstruction_in_progress) {
            // Reconstruction is in progress on server
            if (!g_reconstruction_state.reconstruction_in_progress.load()) {
                g_reconstruction_state.reconstruction_in_progress.store(true);
                log_worker_state("RECONSTRUCTION_IN_PROGRESS",
                                true, false, epoch_info.reconstruction_id, 0,
                                g_reconstruction_state.total_queries.load());
            }
        } else {
            // Server says reconstruction_in_progress = false but epoch hasn't changed
            // This can happen if reconstruction failed or was cancelled
            // Clear the local flag to unblock search threads
            if (g_reconstruction_state.reconstruction_in_progress.load()) {
                g_reconstruction_state.reconstruction_in_progress.store(false);
                log_worker_state("RECONSTRUCTION_CLEARED_NO_EPOCH_CHANGE",
                                false, false, epoch_info.reconstruction_id, 0,
                                g_reconstruction_state.total_queries.load());
            }
        }
    }
}

// Handle reconstruction notification
void handle_reconstruction_notification(const ReconstructionState& state) {
    std::cout << "[Worker " << FLAGS_client_id << "] Received reconstruction notification" << std::endl;
    std::cout << "  Phase: " << static_cast<int>(state.phase) << std::endl;
    std::cout << "  Progress: " << state.progress << "%" << std::endl;
    std::cout << "  Message: " << state.message << std::endl;
    
    if (state.phase == ReconstructionPhase::COMPLETED) {
        g_reconstruction_state.need_update.store(true);
        g_reconstruction_state.reconstruction_in_progress.store(false);
    } else if (state.phase == ReconstructionPhase::BUILDING_INDEX ||
               state.phase == ReconstructionPhase::NOTIFYING_CLIENTS) {
        g_reconstruction_state.reconstruction_in_progress.store(true);
    }
}

// Synchronize insert cache from master via RPC
void sync_insert_cache_from_master(DhnswClient* client) {
    uint64_t current_epoch = g_reconstruction_state.current_epoch.load();

    log_worker_state("SYNCING_INSERT_CACHE",
                    g_reconstruction_state.reconstruction_in_progress.load(),
                    g_reconstruction_state.need_update.load(),
                    g_reconstruction_state.current_reconstruction_id.load(),
                    0, g_reconstruction_state.total_queries.load());

    // Fetch insert cache from server for current epoch
    auto insert_cache = client->GetInsertCache(FLAGS_client_id, current_epoch);

    if (insert_cache.has_cache && insert_cache.vector_count > 0) {
        std::lock_guard<std::mutex> lock(g_reconstruction_state.cache_mutex);
        if (g_reconstruction_state.insert_cache) {
            g_reconstruction_state.insert_cache->clear();
            for (int i = 0; i < insert_cache.vector_count; ++i) {
                const float* vec_ptr = insert_cache.vectors.data() + i * insert_cache.dimension;
                g_reconstruction_state.insert_cache->add(insert_cache.ids[i], vec_ptr);
            }
        }
    }

    int cache_size = 0;
    {
        std::lock_guard<std::mutex> lock(g_reconstruction_state.cache_mutex);
        if (g_reconstruction_state.insert_cache) {
            cache_size = g_reconstruction_state.insert_cache->size();
        }
    }

    log_worker_state("INSERT_CACHE_SYNC_COMPLETE",
                    g_reconstruction_state.reconstruction_in_progress.load(),
                    g_reconstruction_state.need_update.load(),
                    g_reconstruction_state.current_reconstruction_id.load(),
                    cache_size, g_reconstruction_state.total_queries.load());
}

// Apply insert cache to search results
void apply_insert_cache_to_results(
    const float* query,
    int dim,
    std::vector<float>& distances,
    std::vector<dhnsw_idx_t>& labels,
    int top_k) {
    
    std::lock_guard<std::mutex> lock(g_reconstruction_state.cache_mutex);
    
    if (!g_reconstruction_state.insert_cache || 
        g_reconstruction_state.insert_cache->size() == 0) {
        return;
    }
    
    // Search in the cache
    auto cache_results = g_reconstruction_state.insert_cache->search(query, top_k);
    
    // Merge with existing results
    // This is a simplified merge - in production you'd do a proper k-way merge
    for (const auto& [id, dist] : cache_results) {
        for (int i = 0; i < top_k; ++i) {
            if (dist < distances[i]) {
                // Insert here and shift others
                for (int j = top_k - 1; j > i; --j) {
                    distances[j] = distances[j-1];
                    labels[j] = labels[j-1];
                }
                distances[i] = dist;
                labels[i] = id;
                break;
            }
        }
    }
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    dhnsw::load_dataset_config();
    const auto& cfg = dhnsw::GlobalDatasetConfig;
    
    // Initialize reconstruction state
    g_reconstruction_state.insert_cache = std::make_unique<LSHCache>(cfg.dim);
    g_reconstruction_state.handler = std::make_unique<ClientReconstructionHandler>(
        FLAGS_client_id, false /* not master */);
    g_reconstruction_state.handler->set_update_callback(handle_reconstruction_notification);
    g_reconstruction_state.throughput_logger = std::make_unique<ThroughputLogger>(
        FLAGS_throughput_log, FLAGS_client_id);
    g_reconstruction_state.throughput_logger->start_logging(100);

    // Print configuration
    std::cout << "=== Worker Search Node Configuration ===" << std::endl;
    std::cout << "Client ID: " << FLAGS_client_id << std::endl;
    std::cout << "Dataset: " << dhnsw::FLAGS_dataset << std::endl;
    std::cout << "Dimension: " << cfg.dim << std::endl;
    std::cout << "Batch size: " << cfg.batch_size << std::endl;
    std::cout << "==========================================" << std::endl << std::endl;
    
    // Read dataset
    std::string query_data_path = cfg.query_data_path;
    std::string ground_truth_path = cfg.ground_truth_path;
    int dim = cfg.dim;
    int num_threads = cfg.num_threads;
    std::vector<int> ef_search_values = cfg.ef_search_values;
    
    std::vector<float> query_data_tmp;
    if (query_data_path.find("bvecs") != std::string::npos) {
        query_data_tmp = read_bvecs(query_data_path, dim_query_data, n_query_data);
    } else {
        query_data_tmp = read_fvecs(query_data_path, dim_query_data, n_query_data);
    }
    std::vector<int> ground_truth_tmp = read_ivecs(ground_truth_path, dim_ground_truth, n_query_data);
    
    // Sample query data
    n_query_data = 0;
    for (int i = 0; i < (int)query_data_tmp.size() / dim_query_data; i++) {
        if (i % cfg.sampling_mod == cfg.sampling_count) {
            query_data.insert(query_data.end(), 
                              query_data_tmp.begin() + i * dim_query_data,
                              query_data_tmp.begin() + (i + 1) * dim_query_data);
            ground_truth.insert(ground_truth.end(),
                                ground_truth_tmp.begin() + i * dim_ground_truth,
                                ground_truth_tmp.begin() + (i + 1) * dim_ground_truth);
            n_query_data++;
        }
    }
    
    int original_n_query_data = n_query_data;
    std::vector<float> original_query_data = query_data;
    std::vector<int> original_ground_truth = ground_truth;
    for (int rep = 1; rep < cfg.num_reps; rep++) {
        query_data.insert(query_data.end(), original_query_data.begin(), original_query_data.end());
        ground_truth.insert(ground_truth.end(), original_ground_truth.begin(), original_ground_truth.end());
    }
    n_query_data = original_n_query_data * cfg.num_reps;
    
    int queries_per_thread = n_query_data / num_threads;
    ground_truth.resize(n_query_data * dim_ground_truth);
    
    int omp_threads_per_worker = FLAGS_physical_cores_per_thread;
    
    std::vector<pthread_t> threads(num_threads);
    std::vector<thread_param_t> thread_params(num_threads);
    
    // Initialize RDMA resources
    ConnectManager cm(FLAGS_rdma_server_address);
    if (cm.wait_ready(1000000, 20) == IOCode::Timeout) {
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
        std::string qp_name = "-" + FLAGS_client_id + "@" + std::to_string(timestamp) + std::to_string(i);
        auto qp_res = cm.cc_rc(qp_name, qps[i], FLAGS_reg_nic_name, QPConfig());
        RDMA_ASSERT(qp_res == IOCode::Ok) << std::get<0>(qp_res.desc);
        keys[i] = std::get<1>(qp_res.desc);
        
        size_t fixed_size = 1UL * 1024 * 1024 * 1024; 
        local_mems[i] = Arc<RMem>(new RMem(fixed_size));
        local_mrs[i] = RegHandler::create(local_mems[i], nics[i]).value();
    }
    
    // Set core assignments
    for (int i = 0; i < num_threads; ++i) {
        thread_params[i].core_start = nth_phys_core_on_numa1(i * omp_threads_per_worker);
        if(i >= 4) {
            thread_params[i].core_start = (i-4) * omp_threads_per_worker * 2;
        }
    }

    // Create gRPC client for status poller
    grpc::ChannelArguments poller_args;
    poller_args.SetMaxReceiveMessageSize(INT_MAX);
    poller_args.SetMaxSendMessageSize(INT_MAX);
    std::shared_ptr<grpc::Channel> poller_channel = grpc::CreateCustomChannel(
        FLAGS_server_address, grpc::InsecureChannelCredentials(), poller_args);
    DhnswClient* poller_client = new DhnswClient(poller_channel);

    // Start status poller thread
    std::thread poller_thread(reconstruction_status_poller, poller_client);
    poller_thread.detach();

    // Initialize last_refresh_timestamp_ms to current time to prevent immediate fallback refresh
    // (fallback refresh triggers when now - last_refresh >= 60s, so setting to current time prevents it)
    g_reconstruction_state.last_refresh_timestamp_ms.store(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    
    // Initialize worker count for ACK coordination
    // All workers must refresh before sending ACK to server
    g_reconstruction_state.num_workers.store(num_threads);
    std::cout << "[Main] Initialized with " << num_threads << " worker threads for ACK coordination" << std::endl;
    
    // Create worker threads
    for (int i = 0; i < num_threads; ++i) {
        thread_params[i].thread_id = i;
        thread_params[i].latency = 0;
        thread_params[i].omp_threads_per_worker = omp_threads_per_worker;
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
    
    // Output statistics
    std::cout << "\n=== Final Statistics ===" << std::endl;
    std::cout << "Total queries: " << g_reconstruction_state.total_queries.load() << std::endl;
    std::cout << "Total blocked time: " << g_reconstruction_state.blocked_duration_us.load() << " us" << std::endl;
    
    // Aggregate results
    std::vector<float> avg_recalls;
    std::vector<double> avg_latencies;
    std::vector<double> avg_network_latency;    
    std::vector<double> avg_duration_meta_search;
    std::vector<double> avg_compute_time;
    std::vector<double> avg_deserialize_time;
    std::vector<double> avg_throughput;
    
    for (size_t ef_idx = 0; ef_idx < ef_search_values.size(); ++ef_idx) {
        float sum_recalls = 0.0f;
        double sum_latency = 0.0;
        double sum_network_latency = 0.0;
        double sum_duration_meta_search = 0.0;
        double sum_compute_time = 0.0;
        double sum_deserialize_time = 0.0;
        double sum_throughput = 0.0;
        
        for (int t = 0; t < num_threads; ++t) {
            if (ef_idx < thread_params[t].per_ef_recalls.size()) {
                sum_recalls += thread_params[t].per_ef_recalls[ef_idx];
                sum_latency += thread_params[t].per_ef_latencies[ef_idx];
                sum_network_latency += thread_params[t].per_ef_network_latencies[ef_idx];
                sum_compute_time += thread_params[t].per_ef_compute_times[ef_idx];
                sum_duration_meta_search += thread_params[t].per_ef_duration_meta_search[ef_idx];
                sum_deserialize_time += thread_params[t].per_ef_deserialize_times[ef_idx];
                sum_throughput += thread_params[t].per_ef_throughput[ef_idx];
            }
        }
        avg_recalls.push_back(sum_recalls / num_threads);
        avg_latencies.push_back(sum_latency / num_threads);
        avg_network_latency.push_back(sum_network_latency / num_threads);
        avg_duration_meta_search.push_back(sum_duration_meta_search / num_threads);
        avg_compute_time.push_back(sum_compute_time / num_threads);
        avg_deserialize_time.push_back(sum_deserialize_time / num_threads);
        avg_throughput.push_back(sum_throughput / num_threads);
    }

    // Write results
    std::string results_file = "../benchs/reconstruction/" + FLAGS_client_id + "_results.txt";
    std::ofstream outfile(results_file);
    outfile << "# Worker " << FLAGS_client_id << " Results" << std::endl;
    outfile << "throughput(QPS)\trecall" << std::endl;
    for (size_t i = 0; i < ef_search_values.size(); ++i) {
        outfile << "[" << avg_throughput[i] << ", " << avg_recalls[i] << "]," << std::endl;
    }
    outfile.close();

    std::string details_file = "../benchs/reconstruction/" + FLAGS_client_id + "_details.txt";
    std::ofstream outfile2(details_file);
    outfile2 << "latency(us)\trecall\tnetwork_latency(us)\tcompute_time(us)\t"
             << "meta_search(us)\tdeserialize(us)\tthroughput(QPS)" << std::endl;
    for (size_t i = 0; i < ef_search_values.size(); ++i) {
        outfile2 << "[" << avg_latencies[i] << ", " << avg_recalls[i] << ", " 
                << avg_network_latency[i] << ", " << avg_compute_time[i] << ", " 
                << avg_duration_meta_search[i] << ", " << avg_deserialize_time[i] 
                << ", " << avg_throughput[i] << "]," << std::endl;
    }
    outfile2.close();
    
    // Cleanup
    for (int i = 0; i < num_threads; ++i) {
        std::string qp_name = "-" + FLAGS_client_id + "@" + std::to_string(timestamp) + std::to_string(i);
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
    
    std::vector<int> ef_search_values = dhnsw::GlobalDatasetConfig.ef_search_values;
    
    // Initialize LocalHnsw
    int dim = dhnsw::GlobalDatasetConfig.dim; 
    int num_sub_hnsw = dhnsw::GlobalDatasetConfig.num_sub_hnsw;      
    int meta_hnsw_neighbors = dhnsw::GlobalDatasetConfig.meta_hnsw_neighbors;
    int sub_hnsw_neighbors = dhnsw::GlobalDatasetConfig.sub_hnsw_neighbors;
    
    grpc::ChannelArguments args;
    args.SetMaxReceiveMessageSize(INT_MAX);
    args.SetMaxSendMessageSize(INT_MAX);

    std::shared_ptr<grpc::Channel> channel = grpc::CreateCustomChannel(
        FLAGS_server_address, grpc::InsecureChannelCredentials(), args);

    DhnswClient* dhnsw_client = new DhnswClient(channel);
    LocalHnsw local_hnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, dhnsw_client);
    local_hnsw.init();
    
    std::cout << "Thread " << thread_id << " initialized, mapping size: " 
              << local_hnsw.get_local_mapping().size() << std::endl;
    
#if DISABLE_RUNTIME_OFFSET_SYNC
    std::cout << "[Worker " << thread_id << "] DEBUG: DISABLE_RUNTIME_OFFSET_SYNC=1 - "
              << "Offsets frozen at epoch " << local_hnsw.get_current_epoch() 
              << ", will NOT update during search" << std::endl;
#endif
    
    local_hnsw.set_rdma_qp(thread_param.qp, thread_param.remote_attr, thread_param.local_mr);
    local_hnsw.set_remote_attr(thread_param.remote_attr);
    local_hnsw.set_local_mr(thread_param.local_mr, thread_param.local_mem);

    PipelinedSearchManager search_manager(&local_hnsw, thread_param.core_start, 
                                         thread_param.omp_threads_per_worker);
    
    // Register the PipelinedSearchManager with LocalHnsw for synchronization during init()
    // This allows init() to wait for in-flight batches to complete before updating state
    local_hnsw.set_pipelined_search_manager(&search_manager);

    int query_start = thread_param.query_start;
    int query_end = thread_param.query_end;
    int n_query_data_thread = query_end - query_start;
    const float* query_data_ptr = query_data.data() + query_start * dim_query_data;
    const int* ground_truth_ptr = ground_truth.data() + query_start * dim_ground_truth;
    
    auto run_ef_benchmark = [&](int ef, size_t duration_sec) {
        int batch_size = dhnsw::GlobalDatasetConfig.batch_size;
        int top_k = 1;
        int branching_k = 5;
        int queries_executed = 0;
        double total_compute_time = 0.0;
        double total_network_latency = 0.0;
        double total_meta_search_time = 0.0;
        double total_batch_time = 0.0;
        double total_deserialize_time = 0.0;
        
        std::vector<int> all_retrieved;
        std::vector<int> all_ground_truth;
        
        float* batch_meta_distances = new float[branching_k * batch_size];
        dhnsw_idx_t* batch_meta_labels = new dhnsw_idx_t[branching_k * batch_size];
        dhnsw_idx_t* batch_sub_hnsw_tags = new dhnsw_idx_t[top_k * batch_size];
        dhnsw_idx_t* batch_labels = new dhnsw_idx_t[top_k * batch_size];
        float* batch_distances = new float[top_k * batch_size];
        dhnsw_idx_t* batch_original_index = new dhnsw_idx_t[top_k * batch_size];
        
        auto bench_start = high_resolution_clock::now();
        int query_index = 0;
        int batch_id = 0;
        
        // Track for fallback periodic refresh (30 seconds - much less frequent than before)
        auto last_refresh_time = high_resolution_clock::now();
        constexpr int FALLBACK_REFRESH_INTERVAL_MS = 60000;  // 60 seconds fallback
        
        // Track consecutive errors - triggers refresh if stale data detected
        int consecutive_errors = 0;
        constexpr int MAX_CONSECUTIVE_ERRORS = 3;
        
        while (duration_cast<seconds>(high_resolution_clock::now() - bench_start).count() < (long)duration_sec) {
#if DISABLE_RUNTIME_OFFSET_SYNC
            // =================================================================
            // DEBUG: Runtime offset synchronization is DISABLED.
            // Offsets are frozen from initial reconstruction and will NOT be
            // updated during normal search operations.
            // This isolates whether segfaults are caused by epoch-based updates.
            // To re-enable: set DISABLE_RUNTIME_OFFSET_SYNC to 0 in DistributedHnsw.h
            // =================================================================
            
            // Clear any pending need_update flags to prevent confusion in logs
            // but do NOT actually call init() or update offsets
            if (g_reconstruction_state.need_update.load()) {
                g_reconstruction_state.need_update.store(false);
                std::cout << "[Worker " << thread_id << "] DEBUG: Epoch change detected but runtime offset sync is DISABLED" << std::endl;
            }
#else
            // Check if this worker needs to refresh to new epoch
            // IMPORTANT: Each worker has its OWN LocalHnsw that needs independent refresh
            // We use refresh_epoch to track which epoch workers should refresh to
            uint64_t target_epoch = g_reconstruction_state.refresh_epoch.load();
            uint64_t my_epoch = local_hnsw.get_current_epoch();
            
            if (target_epoch > 0 && my_epoch < target_epoch) {
                // This worker needs to refresh to the new epoch
                std::cout << "[Worker " << thread_id << "] Refreshing from epoch " 
                          << my_epoch << " to " << target_epoch << "..." << std::endl;
                
                try {
                    // init() waits for in-flight batch to complete before updating state
                    // This is safe because we hold processing_lock inside init()
                    local_hnsw.init();
                    
                    uint64_t new_epoch = local_hnsw.get_current_epoch();
                    std::cout << "[Worker " << thread_id << "] Refreshed to epoch " << new_epoch << std::endl;
                    
                    // Signal that this worker has refreshed
                    // If this is the last worker, send ACK to server
                    {
                        std::lock_guard<std::mutex> lock(g_reconstruction_state.refresh_mutex);
                        int refreshed = g_reconstruction_state.workers_refreshed.fetch_add(1) + 1;
                        int total = g_reconstruction_state.num_workers.load();
                        
                        std::cout << "[Worker " << thread_id << "] Refresh complete ("
                                  << refreshed << "/" << total << " workers)" << std::endl;
                        
                        if (refreshed >= total && !g_reconstruction_state.ack_sent.load()) {
                            // This is the last worker - send ACK to server
                            g_reconstruction_state.ack_sent.store(true);
                            
                            // Get the client to send ACK
                            // Note: We use dhnsw_client from this thread
                            dhnsw_client->AcknowledgeReconstruction(
                                FLAGS_client_id, 
                                g_reconstruction_state.current_reconstruction_id.load(),
                                new_epoch);
                            
                            std::cout << "[Worker " << thread_id << "] All workers refreshed, ACK sent to server" << std::endl;
                            
                            // Update global timestamp
                            g_reconstruction_state.last_refresh_timestamp_ms.store(
                                duration_cast<milliseconds>(high_resolution_clock::now().time_since_epoch()).count());
                            g_reconstruction_state.need_update.store(false);
                        }
                    }
                } catch (const std::exception& e) {
                    std::cerr << "[Worker " << thread_id << "] Failed to refresh: " << e.what() << std::endl;
                    // Don't count failed refreshes - will retry on next iteration
                }
                
                last_refresh_time = high_resolution_clock::now();
            }
#endif
            
            int current_batch_size = std::min(batch_size, 
                n_query_data_thread - (query_index % n_query_data_thread));
            if (current_batch_size <= 0) {
                current_batch_size = batch_size;
            }
            
            const float* batch_query_data_ptr = query_data_ptr + 
                ((query_index % n_query_data_thread) * dim_query_data);
            
            // Record batch start time
            auto batch_start_time = high_resolution_clock::now();
            uint64_t batch_start_ms = duration_cast<milliseconds>(
                batch_start_time.time_since_epoch()).count();
            
            // Meta-search
            std::vector<int> sub_hnsw_tosearch_batch;
            std::unordered_map<int, std::unordered_set<int>> searchset;
            double meta_duration = 0;
            double sub_duration = 0;
            std::tuple<double, double, double> batch_result{0, 0, 0};
            
            try {
                auto meta_start = high_resolution_clock::now();
                searchset = local_hnsw.meta_search_pipelined(current_batch_size, batch_query_data_ptr, 
                    branching_k, batch_meta_distances, batch_meta_labels, sub_hnsw_tosearch_batch, 
                    thread_param.core_start, thread_param.omp_threads_per_worker);
                auto meta_end = high_resolution_clock::now();
                meta_duration = duration_cast<microseconds>(meta_end - meta_start).count();
                total_meta_search_time += meta_duration;
                
                // Sub-search
                std::fill(batch_sub_hnsw_tags, batch_sub_hnsw_tags + top_k * current_batch_size, -1);
                auto sub_start = high_resolution_clock::now();
                
                batch_result = search_manager.process_batch(
                    current_batch_size, batch_query_data_ptr, top_k,
                    batch_distances, batch_labels, searchset, batch_sub_hnsw_tags, ef);
                
                auto sub_end = high_resolution_clock::now();
                sub_duration = duration_cast<microseconds>(sub_end - sub_start).count();
            } catch (const std::exception& e) {
                // Handle RDMA/deserialization errors - likely stale offsets
                // Don't block, just log and continue - will refresh offsets on next iteration
                consecutive_errors++;
                std::cerr << "[Worker " << thread_id << "] Search error (" << consecutive_errors 
                          << "/" << MAX_CONSECUTIVE_ERRORS << "): " << e.what() << std::endl;
                
#if DISABLE_RUNTIME_OFFSET_SYNC
                // DEBUG: Do NOT trigger offset refresh on errors - offsets are frozen
                // If offsets are stale, search will continue with stale data (ANN is approximate)
#else
                // Force immediate offset refresh on next loop iteration
                g_reconstruction_state.need_update.store(true);
#endif
                continue;  // Skip this batch, try next
            }
            
            // Record batch end time
            auto batch_end_time = high_resolution_clock::now();
            uint64_t batch_end_ms = duration_cast<milliseconds>(
                batch_end_time.time_since_epoch()).count();
            
            // Record batch to throughput logger
            g_reconstruction_state.throughput_logger->record_batch(
                thread_id, batch_start_ms, batch_end_ms, 
                current_batch_size, 0, false);
            
            // Reset error counter on success
            consecutive_errors = 0;
            total_compute_time += std::get<0>(batch_result);
            total_network_latency += std::get<1>(batch_result);
            total_deserialize_time += std::get<2>(batch_result);
            total_batch_time += (meta_duration + sub_duration);
            
            // Apply insert cache results (if any pending inserts)
            if (g_reconstruction_state.insert_cache && 
                g_reconstruction_state.insert_cache->size() > 0) {
                for (int i = 0; i < current_batch_size; i++) {
                    std::vector<float> q_dists(top_k);
                    std::vector<dhnsw_idx_t> q_labels(top_k);
                    for (int k = 0; k < top_k; k++) {
                        q_dists[k] = batch_distances[i * top_k + k];
                        q_labels[k] = batch_labels[i * top_k + k];
                    }
                    apply_insert_cache_to_results(
                        batch_query_data_ptr + i * dim_query_data,
                        dim,
                        q_dists,
                        q_labels,
                        top_k);
                    // Copy back
                    for (int k = 0; k < top_k; k++) {
                        batch_distances[i * top_k + k] = q_dists[k];
                        batch_labels[i * top_k + k] = q_labels[k];
                    }
                }
            }
            
            // Log batch statistics
            bench::Statics batch_stat;
            batch_stat.set_batch_metrics(
                std::get<0>(batch_result), std::get<1>(batch_result), std::get<2>(batch_result),
                meta_duration,
                (1e6 * current_batch_size) / (meta_duration + sub_duration),
                (meta_duration + sub_duration),
                batch_id);      
            bench::Reporter::report_batch(batch_stat, batch_id, FLAGS_log_file);
            
            // Compute original indices
            auto mapping = local_hnsw.get_local_mapping();
            for (int i = 0; i < current_batch_size; i++) {
                int pos = i * top_k;
                if (batch_sub_hnsw_tags[pos] >= 0 && 
                    batch_sub_hnsw_tags[pos] < (dhnsw_idx_t)mapping.size() &&
                    batch_labels[pos] >= 0 && 
                    batch_labels[pos] < (dhnsw_idx_t)mapping[batch_sub_hnsw_tags[pos]].size()) {
                    batch_original_index[pos] = mapping[batch_sub_hnsw_tags[pos]][batch_labels[pos]];
                } else {
                    batch_original_index[pos] = -1;
                }
            }
            
            // Accumulate results
            for (int i = 0; i < current_batch_size; i++) {
                int pos = i * top_k;
                // Use modulo to wrap around ground truth access, matching query data wrapping
                int gt_index = ((query_index + i) % n_query_data_thread) * dim_ground_truth;
                int gt = *(ground_truth_ptr + gt_index);
                int retrieved = batch_original_index[pos];
                all_ground_truth.push_back(gt);
                all_retrieved.push_back(retrieved);
            }
            
            queries_executed += current_batch_size;
            g_reconstruction_state.total_queries.fetch_add(current_batch_size);
            
            query_index += current_batch_size;
            batch_id++;
        }
        
        // Cleanup
        delete[] batch_meta_distances;
        delete[] batch_meta_labels;
        delete[] batch_sub_hnsw_tags;
        delete[] batch_labels;
        delete[] batch_distances;   
        delete[] batch_original_index;
        
        // Calculate recall
        int total_correct = 0;
        for (size_t i = 0; i < all_retrieved.size(); i++) {
            if (all_retrieved[i] == all_ground_truth[i]) {
                total_correct++;
            }
        }
        float recall = (all_retrieved.size() > 0) ? 
            static_cast<float>(total_correct) / all_retrieved.size() : 0.0f;
        
        double avg_total_latency = (queries_executed > 0) ? total_batch_time / queries_executed : 0.0;
        double avg_meta = (queries_executed > 0) ? total_meta_search_time / queries_executed : 0.0;
        double avg_compute = (queries_executed > 0) ? total_compute_time / queries_executed : 0.0;
        double avg_network = (queries_executed > 0) ? total_network_latency / queries_executed : 0.0;
        double avg_deserialize = (queries_executed > 0) ? total_deserialize_time / queries_executed : 0.0;
        double throughput = queries_executed / (avg_total_latency * 1e-6);
        
        std::cout << "Thread " << thread_id << " EF " << ef << " benchmark:" << std::endl;
        std::cout << "  Queries executed: " << queries_executed << std::endl;
        std::cout << "  Avg latency: " << avg_total_latency << " us" << std::endl;
        std::cout << "  Recall: " << recall << std::endl;
        std::cout << "  Throughput: " << throughput << " QPS" << std::endl;
        
        thread_param.per_ef_latencies.push_back(avg_total_latency);
        thread_param.per_ef_recalls.push_back(recall);
        thread_param.per_ef_network_latencies.push_back(avg_network);
        thread_param.per_ef_compute_times.push_back(avg_compute);
        thread_param.per_ef_duration_meta_search.push_back(avg_meta);
        thread_param.per_ef_deserialize_times.push_back(avg_deserialize);
        thread_param.per_ef_throughput.push_back(throughput);
    };
    
    // Run benchmarks
    for (int ef : ef_search_values) {
        std::cout << "=== Thread " << thread_id << " testing EF = " << ef 
                  << " for " << FLAGS_benchmark_duration << " seconds ===" << std::endl;
        run_ef_benchmark(ef, FLAGS_benchmark_duration);
    }

    delete dhnsw_client;
    return nullptr;
}

