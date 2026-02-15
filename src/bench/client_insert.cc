// client_multithreads_benchmark.cc

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
#include "../util/read_dataset.h"

typedef int64_t dhnsw_idx_t;

DEFINE_string(server_address, "0.0.0.0:50051", "Address of the gRPC server.");
DEFINE_string(rdma_server_address, "192.168.1.2:8888", "Address of the RDMA server (InfiniBand IP).");
DEFINE_int32(use_nic_idx, 3, "Which NIC to create QP");
DEFINE_int32(reg_nic_name, 73, "The name to register an opened NIC at rctrl.");
DEFINE_int32(reg_mem_name, 73, "The name to register an MR at rctrl.");
DEFINE_string(query_data_path, "../datasets/sift/sift_query.fvecs", "Path to the query data.");
DEFINE_string(ground_truth_path, "../datasets/sift/sift_groundtruth.ivecs", "Path to the ground truth data.");
// DEFINE_string(query_data_path, "../datasets/gist/gist_query.fvecs", "Path to the query data.");
// DEFINE_string(ground_truth_path, "../datasets/gist/gist_groundtruth.ivecs", "Path to the ground truth data.");

DEFINE_int32(benchmark_duration, 1, "Duration (in seconds) to run each ef benchmark.");
DEFINE_int32(physical_cores_per_thread, 36, "Number of physical cores per thread");
DEFINE_bool(use_physical_cores_only, true, "Whether to use only physical cores (skip hyperthreads)");
DEFINE_string(log_file, "../benchs/pipeline/test/test.log", "Path to the log file.");
DEFINE_int32(insert_percentage, 10, "Percentage of insert operations (0-100), remaining will be query operations");
DEFINE_int32(batch_size, 8000, "Batch size for operations (recommended: 1000-5000, max: 8000)");
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

// Global dataset variables
std::vector<float> query_data;
std::vector<int> ground_truth;
int dim_query_data;
int n_query_data;
int dim_ground_truth;
// GlobalTracer g_tracer;
// Thread parameters and benchmark metrics per ef value
struct thread_param_t {
    int thread_id;
    int omp_threads_per_worker;
    int core_start;
    double latency;
    int query_start;
    int query_end;
    // Collected metrics for each tested ef value:
    std::vector<float> per_ef_recalls;
    std::vector<double> per_ef_latencies;
    std::vector<double> per_ef_network_latencies;
    std::vector<double> per_ef_duration_meta_search;
    std::vector<double> per_ef_compute_times;
    std::vector<double> per_ef_deserialize_times;
    std::vector<double> per_ef_throughput;
    // RDMA and other resources:
    Arc<RMem> local_mem;
    Arc<RegHandler> local_mr;
    Arc<RNic> nic;
    std::shared_ptr<RC> qp;
    rmem::RegAttr remote_attr;
    uint64_t key;
};

void* client_worker(void* param);

void bind_thread_to_cores(int thread_id, int core_start, int cores_per_thread, bool physical_cores_only) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    if (physical_cores_only) {
        
        for (int i = 0; i < cores_per_thread; i++) {
            int physical_core = core_start + (i * 2); // Skip hyperthreads by using only even-numbered cores
            CPU_SET(physical_core, &cpuset);
        }

        std::cout << "Thread " << thread_id << " bound to physical cores starting at " 
                  << core_start << " (skipping hyperthreads)" << std::endl;
    } else {
        // Original behavior - use all cores including hyperthreads
        for (int i = 0; i < cores_per_thread; i++) {
            CPU_SET(core_start + i, &cpuset);
        }
        std::cout << "Thread " << thread_id << " bound to cores " 
                  << core_start << "-" << (core_start + cores_per_thread - 1) << std::endl;
    }
    
    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error: unable to set thread affinity for thread " << thread_id 
                  << ", error code: " << rc << std::endl;
    }
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // --- Print workload configuration ---
    int query_percentage = 100 - FLAGS_insert_percentage;
    std::cout << "=== Workload Configuration ===" << std::endl;
    std::cout << "Insert percentage: " << FLAGS_insert_percentage << "%" << std::endl;
    std::cout << "Query percentage: " << query_percentage << "%" << std::endl;
    std::cout << "Batch size: " << FLAGS_batch_size << std::endl;
    std::cout << "==============================" << std::endl << std::endl;

    // --- Read dataset ---
    std::string query_data_path = FLAGS_query_data_path;
    std::string ground_truth_path = FLAGS_ground_truth_path;
    std::vector<float> query_data_tmp = read_fvecs(query_data_path, dim_query_data, n_query_data);
    std::vector<int> ground_truth_tmp = read_ivecs(ground_truth_path, dim_ground_truth, n_query_data);
    
    // Sample 1/3 of the query data (and corresponding ground truth)
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
    std::cout << "queries_per_thread: " << queries_per_thread << std::endl;
    ground_truth.resize(n_query_data * dim_ground_truth);
    
    // Calculate core assignments for physical cores
    int omp_threads_per_worker = FLAGS_physical_cores_per_thread;
    if (FLAGS_use_physical_cores_only) {
        std::cout << "Using physical cores only (skipping hyperthreads)" << std::endl;
    } else {
        std::cout << "Using all cores including hyperthreads" << std::endl;
    }
    
    std::vector<pthread_t> threads(num_threads);
    std::vector<thread_param_t> thread_params(num_threads);
    
    // --- Initialize RDMA resources in main thread ---
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
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    for (int i = 0; i < num_threads; ++i) {
        nics[i] = RNic::create(RNicInfo::query_dev_names().at(FLAGS_use_nic_idx)).value();
        qps[i] = RC::create(nics[i], QPConfig()).value();
        std::string qp_name = "-client@" + std::to_string(timestamp) + std::to_string(i);
        auto qp_res = cm.cc_rc(qp_name, qps[i], FLAGS_reg_nic_name, QPConfig());
        RDMA_ASSERT(qp_res == IOCode::Ok) << std::get<0>(qp_res.desc);
        keys[i] = std::get<1>(qp_res.desc);
        RDMA_LOG(4) << "Thread " << i << " fetch QP authentication key: " << keys[i];
  
        size_t fixed_size = 2UL * 1024 * 1024 * 1024;  // 2GB
        local_mems[i] = Arc<RMem>(new RMem(fixed_size));
        local_mrs[i] = RegHandler::create(local_mems[i], nics[i]).value();
        RDMA_LOG(4) << "Thread " << i << " RDMA resources initialized";
    }
    
    // --- Create worker threads ---
    for (int i = 0; i < num_threads; ++i) {
        thread_params[i].thread_id = i;
        thread_params[i].latency = 0;
        thread_params[i].omp_threads_per_worker = omp_threads_per_worker;
        
        // Calculate core start position based on physical core selection strategy
        if (FLAGS_use_physical_cores_only) {
            // If using physical cores only, space out the core assignments
            if(i < 4) {
                thread_params[i].core_start = i * omp_threads_per_worker * 2; // Multiply by 2 to skip hyperthreads
            } else {
                thread_params[i].core_start = (i-4) * omp_threads_per_worker * 2 + 1; // Multiply by 2 to skip hyperthreads
            }
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
    
    for (int i = 0; i < num_threads; ++i) {
        void* status;
        int rc = pthread_join(threads[i], &status);
        if (rc) {
            std::cerr << "Error: unable to join, " << rc << std::endl;
            exit(-1);
        }
    }
    
    // std::vector<int> ef_search_values = {1, 1, 1, 2, 4, 8, 16, 24, 32, 40, 48, 48, 48};
    std::vector<int> ef_search_values = {48, 48, 48};
   
    std::vector<int> ef_values;
    std::vector<float> avg_recalls;
    std::vector<double> avg_latencies;
    std::vector<double> avg_network_latency;    
    std::vector<double> avg_duration_meta_search;
    std::vector<double> avg_compute_time;
    std::vector<double> avg_deserialize_time;
    std::vector<double> avg_throughput;
    for (size_t ef_idx = 0; ef_idx < ef_search_values.size(); ++ef_idx) {
        int ef = ef_search_values[ef_idx];
        ef_values.push_back(ef);
        float sum_recalls = 0.0f;
        double sum_latency = 0.0;
        double sum_network_latency = 0.0;
        double sum_duration_meta_search = 0.0;
        double sum_compute_time = 0.0;
        double sum_deserialize_time = 0.0;
        double sum_throughput = 0.0;
        std::cout << "EF: " << ef << std::endl;
        for (int t = 0; t < num_threads; ++t) {
            std::cout << "Thread " << t << " recall: " << thread_params[t].per_ef_recalls[ef_idx] << std::endl;
            std::cout << "Thread " << t << " latency: " << thread_params[t].per_ef_latencies[ef_idx] << std::endl;
            std::cout << "Thread " << t << " network latency: " << thread_params[t].per_ef_network_latencies[ef_idx] << std::endl;
            std::cout << "Thread " << t << " duration_meta_search: " << thread_params[t].per_ef_duration_meta_search[ef_idx] << std::endl;
            std::cout << "Thread " << t << " compute time: " << thread_params[t].per_ef_compute_times[ef_idx] << std::endl;
            sum_recalls += thread_params[t].per_ef_recalls[ef_idx];
            sum_latency += thread_params[t].per_ef_latencies[ef_idx];
            sum_network_latency += thread_params[t].per_ef_network_latencies[ef_idx];
            sum_compute_time += thread_params[t].per_ef_compute_times[ef_idx];
            sum_duration_meta_search += thread_params[t].per_ef_duration_meta_search[ef_idx];
            sum_deserialize_time += thread_params[t].per_ef_deserialize_times[ef_idx];
            sum_throughput += thread_params[t].per_ef_throughput[ef_idx];
        }
        avg_recalls.push_back(sum_recalls / num_threads);
        avg_latencies.push_back(sum_latency / num_threads);
        avg_network_latency.push_back(sum_network_latency / num_threads);
        avg_duration_meta_search.push_back(sum_duration_meta_search / num_threads);
        avg_compute_time.push_back(sum_compute_time / num_threads);
        avg_deserialize_time.push_back(sum_deserialize_time / num_threads);
        avg_throughput.push_back(sum_throughput / num_threads);
    }

    // Write results to output file
    std::ofstream outfile("../benchs/pipeline/test/sift1M@1benchmark_results_test.txt");
    outfile << "# Mixed workload: " << FLAGS_insert_percentage << "% insert, " << query_percentage << "% query" << std::endl;
    outfile << "throughput(ops/s)\t recall" << std::endl;
    for (size_t i = 0; i < ef_values.size(); ++i) {
        outfile << "[" <<  avg_throughput[i] << ", " << avg_recalls[i] <<"],"<< std::endl;
    }
    outfile.close();

    std::ofstream outfile2("../benchs/pipeline/test/sift1M@1benchmark_details.txt");
    outfile2 << "# Mixed workload: " << FLAGS_insert_percentage << "% insert, " << query_percentage << "% query" << std::endl;
    outfile2 << "latency(us)\trecall\tnetwork_latency(us)\tcompute_time(us)\tduration_meta_search(us)\tdeserialize_time(us)\tthroughput(ops/s)" << std::endl;
    for (size_t i = 0; i < ef_values.size(); ++i) {
        outfile2 << "[" <<  avg_latencies[i] << ", " << avg_recalls[i] <<", " << avg_network_latency[i] << ", " << avg_compute_time[i] << ", " << avg_duration_meta_search[i] << ", " << avg_deserialize_time[i] << ", " << avg_throughput[i] <<"],"<< std::endl;
    }
    outfile2.close();
    // --- Clean up RDMA resources ---
    for (int i = 0; i < num_threads; ++i) {
        std::string qp_name = "-client@" + std::to_string(timestamp) + std::to_string(i);
        auto del_res = cm.delete_remote_rc(qp_name, thread_params[i].key);
        RDMA_ASSERT(del_res == IOCode::Ok)
            << "Thread " << i << " delete remote QP error: " << del_res.desc;
        qps[i].reset();
    }
    return 0;
}

void* client_worker(void* param) {
    thread_param_t& thread_param = *(thread_param_t*)param;
    int thread_id = thread_param.thread_id;
    bind_thread_to_cores(thread_id, thread_param.core_start, thread_param.omp_threads_per_worker, FLAGS_use_physical_cores_only);
    omp_set_num_threads(thread_param.omp_threads_per_worker);
    
    // --- Define the EF values to test ---
    // std::vector<int> ef_search_values = {1, 1, 1, 2, 4, 8, 16, 24, 32, 40, 48, 48, 48};
    std::vector<int> ef_search_values = {48, 48, 48};
   
    // --- Initialize LocalHnsw ---
    int dim = 128; 
    // int dim = 960;        
    int num_sub_hnsw = 160;      
    // int num_sub_hnsw = 120;      
    int meta_hnsw_neighbors = 16;
    int sub_hnsw_neighbors = 32;
    DhnswClient* dhnsw_client = new DhnswClient(grpc::CreateChannel(FLAGS_server_address, grpc::InsecureChannelCredentials()));
    LocalHnsw local_hnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, dhnsw_client);
    local_hnsw.init();
    std::cout << "Thread " << thread_id << " local_hnsw mapping size: " 
              << local_hnsw.get_local_mapping().size() << std::endl;
    local_hnsw.set_rdma_qp(thread_param.qp, thread_param.remote_attr, thread_param.local_mr);
    local_hnsw.set_remote_attr(thread_param.remote_attr);
    local_hnsw.set_local_mr(thread_param.local_mr, thread_param.local_mem);

    // --- Initialize PipelinedSearchManager for query operations ---
    PipelinedSearchManager search_manager(&local_hnsw, thread_param.core_start, thread_param.omp_threads_per_worker);
    // --- Determine the query data range for this thread ---
    int query_start = thread_param.query_start;
    int query_end = thread_param.query_end;
    int n_query_data_thread = query_end - query_start;
    const float* query_data_ptr = query_data.data() + query_start * dim_query_data;
    const int* ground_truth_ptr = ground_truth.data() + query_start * dim_ground_truth;
    // local_hnsw.start_search_pipeline(1);
    // --- Lambda: Run a timed benchmark in batches and compute recall after the loop ---
    auto run_ef_benchmark = [&](int ef, size_t duration_sec) {
        int batch_size = FLAGS_batch_size;  // Use command line parameter
        int top_k = 1;
        int branching_k = 5;
        int queries_executed = 0;
        int inserts_executed = 0;
        double total_ops_time = 0.0;  // Total time for all operations (insert + query)
        double total_compute_time = 0.0;
        double total_network_latency = 0.0;
        double total_meta_search_time = 0.0;
        double total_deserialize_time = 0.0;
        // Vectors to accumulate results for recall computation
        std::vector<int> all_retrieved;
        std::vector<int> all_ground_truth;
        
        // Allocate arrays for batch processing (query operations)
        float* batch_meta_distances = new float[branching_k * batch_size];
        dhnsw_idx_t* batch_meta_labels = new dhnsw_idx_t[branching_k * batch_size];
        dhnsw_idx_t* batch_sub_hnsw_tags = new dhnsw_idx_t[top_k * batch_size];
        dhnsw_idx_t* batch_labels = new dhnsw_idx_t[top_k * batch_size];
        float* batch_distances = new float[top_k * batch_size];

        auto bench_start = high_resolution_clock::now();
        int query_index = 0;  // index within thread's query range
        int batch_id = 0;
        
        while (duration_cast<seconds>(high_resolution_clock::now() - bench_start).count() < (long)duration_sec) {
            int current_batch_size = std::min(batch_size, n_query_data_thread - (query_index % n_query_data_thread));
            if (current_batch_size <= 0) {
                current_batch_size = batch_size;
            }
            const float* batch_query_data_ptr = query_data_ptr + ((query_index % n_query_data_thread) * dim_query_data);
            
            // Split batch into insert and query portions based on percentage
            int insert_count = (current_batch_size * FLAGS_insert_percentage) / 100;
            int query_count = current_batch_size - insert_count;
            
            auto batch_start = high_resolution_clock::now();
            
            // --- PART 1: INSERT OPERATIONS (first x% of batch) ---
            if (insert_count > 0) {
                const int max_insert_batch = 60;  // Max vectors per insert call
                int remaining = insert_count;
                int offset = 0;
                
                while (remaining > 0) {
                    int current_insert_batch = std::min(max_insert_batch, remaining);
                    const float* insert_data_ptr = batch_query_data_ptr + offset * dim_query_data;
                    std::vector<float> batch_insert_data(insert_data_ptr, 
                                                        insert_data_ptr + current_insert_batch * dim_query_data);
                    std::pair<double, double> batch_result = local_hnsw.insert_to_server(current_insert_batch, batch_insert_data);
                    
                    inserts_executed += current_insert_batch;
                    remaining -= current_insert_batch;
                    offset += current_insert_batch;
                }
            }
            
            // --- PART 2: QUERY OPERATIONS (remaining portion of batch) ---
            if (query_count > 0) {
                const float* query_portion_ptr = batch_query_data_ptr + insert_count * dim_query_data;
                
                // Meta-search for the query portion
                std::vector<int> sub_hnsw_tosearch_batch;
                std::unordered_map<int, std::unordered_set<int>> searchset;
                auto meta_start = high_resolution_clock::now();
                
                searchset = local_hnsw.meta_search_pipelined(query_count, query_portion_ptr, branching_k, 
                                       batch_meta_distances, batch_meta_labels, sub_hnsw_tosearch_batch, 
                                       thread_param.core_start, thread_param.omp_threads_per_worker);
                auto meta_end = high_resolution_clock::now();
                double meta_duration = duration_cast<microseconds>(meta_end - meta_start).count();
                total_meta_search_time += meta_duration;
                
                // Sub-search for the query portion
                std::fill(batch_sub_hnsw_tags, batch_sub_hnsw_tags + top_k * query_count, -1);
                std::tuple<double, double, double> batch_result = search_manager.process_batch(
                    query_count, query_portion_ptr, top_k,
                    batch_distances, batch_labels, searchset, batch_sub_hnsw_tags, ef
                );
                
                total_compute_time += std::get<0>(batch_result);
                total_network_latency += std::get<1>(batch_result);
                total_deserialize_time += std::get<2>(batch_result);
                
                // Collect results for recall computation
                for (int i = 0; i < query_count; ++i) {
                    all_retrieved.push_back(batch_labels[i * top_k]);
                    int gt_idx = ((query_index % n_query_data_thread) + insert_count + i) * dim_ground_truth;
                    all_ground_truth.push_back(ground_truth_ptr[gt_idx]);
                }
                
                queries_executed += query_count;
            }
            
            auto batch_end = high_resolution_clock::now();
            double batch_duration = duration_cast<microseconds>(batch_end - batch_start).count();
            total_ops_time += batch_duration;
            
            query_index += current_batch_size;
            batch_id++;
        }
        
        // Clean up allocated arrays
        delete[] batch_meta_distances;
        delete[] batch_meta_labels;
        delete[] batch_sub_hnsw_tags;
        delete[] batch_labels;
        delete[] batch_distances;
        
        // --- Calculate recall after processing all batches ---
        int total_correct = 0;
        for (size_t i = 0; i < all_retrieved.size(); i++) {
            if (all_retrieved[i] == all_ground_truth[i]) {
                total_correct++;
            }
        }
        float recall = (all_retrieved.size() > 0) ? static_cast<float>(total_correct) / all_retrieved.size() : 0.0f;
        
        // Calculate overall metrics (mixed workload)
        int total_ops = queries_executed + inserts_executed;
        double latency = total_ops_time;
        double throughput = (total_ops > 0) ? total_ops / (total_ops_time * 1e-6) : 0.0;
        
        // Average metrics for query operations only (for breakdown analysis)
        double avg_meta = (queries_executed > 0) ? total_meta_search_time / queries_executed : 0.0;
        double avg_compute = (queries_executed > 0) ? total_compute_time / queries_executed : 0.0;
        double avg_network = (queries_executed > 0) ? total_network_latency / queries_executed : 0.0;
        double avg_deserialize = (queries_executed > 0) ? total_deserialize_time / queries_executed : 0.0;
        
        int query_percentage = 100 - FLAGS_insert_percentage;
        std::cout << "Thread " << thread_id << " EF " << ef << " benchmark:" << std::endl;
        std::cout << "  === Mixed Workload (" << FLAGS_insert_percentage << "% insert, " 
                  << query_percentage << "% query) ===" << std::endl;
        std::cout << "  Total operations: " << total_ops << " (Inserts: " << inserts_executed 
                  << ", Queries: " << queries_executed << ")" << std::endl;
        std::cout << "  latency: " << latency << " us" << std::endl;
        std::cout << "  Overall throughput: " << throughput << " ops/s" << std::endl;
        std::cout << "  Query recall: " << recall << std::endl;
        std::cout << "  --- Query breakdown (avg per query) ---" << std::endl;
        std::cout << "    Meta search: " << avg_meta << " us" << std::endl;
        std::cout << "    Compute time: " << avg_compute << " us" << std::endl;
        std::cout << "    Network latency: " << avg_network << " us" << std::endl;
        std::cout << "    Deserialize time: " << avg_deserialize << " us" << std::endl;
        
        // Save results into thread parameters
        thread_param.per_ef_latencies.push_back(latency);
        thread_param.per_ef_recalls.push_back(recall);
        thread_param.per_ef_network_latencies.push_back(avg_network);
        thread_param.per_ef_compute_times.push_back(avg_compute);
        thread_param.per_ef_duration_meta_search.push_back(avg_meta);
        thread_param.per_ef_deserialize_times.push_back(avg_deserialize);
        thread_param.per_ef_throughput.push_back(throughput);
        

    };
    
    // --- Run benchmark for each EF value ---
    for (int ef : ef_search_values) {
        std::cout << "=== Thread " << thread_id << " testing EF = " << ef 
                  << " for " << FLAGS_benchmark_duration << " seconds ===" << std::endl;
        run_ef_benchmark(ef, FLAGS_benchmark_duration);
    }
    // local_hnsw.stop_search_pipeline();



    delete dhnsw_client;
    // g_tracer.write_to_file("log.csv");
    RDMA_LOG(4) << "Thread " << thread_id << " returns";
    return nullptr;
}