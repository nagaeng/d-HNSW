#include <gflags/gflags.h>
#include <grpcpp/grpcpp.h>

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <cassert>
#include <chrono>
#include <omp.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/sysinfo.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include "../dhnsw/DistributedHnsw.h"
#include "../util/read_dataset.h"
#include "faiss/IndexHNSW.h"
#include <mutex>

// DEFINE_string(dataset_path, "../datasets/sift/sift_base.fvecs", "Path to the dataset.");
DEFINE_string(dataset_path, "../datasets/gist/gist_base.fvecs", "Path to the dataset.");
// DEFINE_string(query_data_path, "../datasets/sift/sift_query.fvecs", "Path to the query data.");
// DEFINE_string(ground_truth_path, "../datasets/sift/sift_groundtruth.ivecs", "Path to the ground truth data.");
DEFINE_string(query_data_path, "../datasets/gist/gist_query.fvecs", "Path to the query data.");
DEFINE_string(ground_truth_path, "../datasets/gist/gist_groundtruth.ivecs", "Path to the ground truth data.");
DEFINE_int32(benchmark_duration, 20, "Duration (in seconds) to run each ef benchmark.");
DEFINE_int32(physical_cores_per_thread, 9, "Number of physical cores per thread");
DEFINE_bool(use_physical_cores_only, true, "Whether to use only physical cores (skip hyperthreads)");

using namespace std;
using namespace std::chrono;

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
    std::vector<double> per_ef_throughput;

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

// Global dataset variables
std::vector<float> query_data;
std::vector<int> ground_truth;
int dim_query_data;
int n_query_data;
int dim_ground_truth;

// Global DHNSW index - properly initialized and thread-safe for read operations
faiss::IndexHNSWFlat* global_dhnsw = nullptr;
std::mutex dhnsw_mutex; // For thread-safe access during initialization

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    // Build the DistributedHnsw index
    int dim = 960;
    // int dim = 128;
    int num_meta = 5000;
    // int num_sub_hnsw = 120;
    int num_sub_hnsw = 80;
    int meta_hnsw_neighbors = 16;
    int sub_hnsw_neighbors = 64;

    std::string base_data_path = FLAGS_dataset_path;
    std::vector<float> base_data;
    int dim_base_data;
    int n_base_data;
    base_data = read_fvecs(base_data_path, dim_base_data, n_base_data);
    std::cout << "Read base data successfully!" << std::endl;

    // Initialize global DHNSW index
    global_dhnsw = new faiss::IndexHNSWFlat(dim, sub_hnsw_neighbors);
    global_dhnsw->hnsw.efConstruction = 120;
    global_dhnsw->add(n_base_data,base_data.data());
    std::cout << "finish build"<< std::endl;
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
    
    int num_threads = 8;
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

      for (int i = 0; i < num_threads; ++i) {
        thread_params[i].thread_id = i;
        thread_params[i].latency = 0;
        thread_params[i].omp_threads_per_worker = omp_threads_per_worker;
        
        // Calculate core start position based on physical core selection strategy
        if (FLAGS_use_physical_cores_only) {
            // If using physical cores only, space out the core assignments
            thread_params[i].core_start = i * omp_threads_per_worker * 2; // Multiply by 2 to skip hyperthreads
        } else {
            thread_params[i].core_start = i * omp_threads_per_worker;
        }
        
        thread_params[i].query_start = i * queries_per_thread;
        thread_params[i].query_end = thread_params[i].query_start + queries_per_thread;
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
    std::vector<double> avg_throughput;
    for (size_t ef_idx = 0; ef_idx < ef_search_values.size(); ++ef_idx) {
        int ef = ef_search_values[ef_idx];
        ef_values.push_back(ef);
        float sum_recalls = 0.0f;
        double sum_latency = 0.0;
        double sum_network_latency = 0.0;
        double sum_duration_meta_search = 0.0;
        double sum_compute_time = 0.0;
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
            sum_throughput += thread_params[t].per_ef_throughput[ef_idx];
        }
        avg_recalls.push_back(sum_recalls / num_threads);
        avg_latencies.push_back(sum_latency / num_threads);
        avg_network_latency.push_back(sum_network_latency / num_threads);
        avg_duration_meta_search.push_back(sum_duration_meta_search / num_threads);
        avg_compute_time.push_back(sum_compute_time / num_threads);
        std::cout<<"avg_compute_time" << sum_compute_time / num_threads << std::endl;
        avg_throughput.push_back(sum_throughput / num_threads);
    }

    // Write results to output file
    std::ofstream outfile("../benchs/pipeline/test/sift1M@1benchmark_results_test.txt");
    outfile << "throughput(QPS)\t recall" << std::endl;
    for (size_t i = 0; i < ef_values.size(); ++i) {
        outfile << "[" <<  avg_throughput[i] << ", " << avg_recalls[i] <<"],"<< std::endl;
    }
    outfile.close();

    std::ofstream outfile2("../benchs/pipeline/test/sift1M@1benchmark_details.txt");
    outfile2 << "latency(us)\trecall\tnetwork_latency(us)\tcompute_time(us)\tduration_meta_search(us)\tthroughput(QPS)" << std::endl;
    for (size_t i = 0; i < ef_values.size(); ++i) {
        outfile2 << "[" <<  avg_latencies[i] << ", " << avg_recalls[i] <<", " << avg_network_latency[i] << ", " << avg_compute_time[i] << ", " << avg_duration_meta_search[i] << ", " << avg_throughput[i] <<"],"<< std::endl;
    }
    outfile2.close();

    // Clean up global resources
    delete global_dhnsw;
    global_dhnsw = nullptr;

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
    // int dim = 128; 
    int dim = 960;        
    int num_sub_hnsw = 80;      
    // int num_sub_hnsw = 120;      
    int meta_hnsw_neighbors = 16;
    int sub_hnsw_neighbors = 64;
    
    // Create a thread-local copy of the global DHNSW index
    faiss::IndexHNSWFlat* local_hnsw_ptr = nullptr;
    {
        std::lock_guard<std::mutex> lock(dhnsw_mutex);
        if (global_dhnsw == nullptr) {
            std::cerr << "Error: global_dhnsw is not initialized!" << std::endl;
            return nullptr;
        }
        local_hnsw_ptr = global_dhnsw; // Share the global index (don't copy)
    }
    faiss::IndexHNSWFlat& local_hnsw = *local_hnsw_ptr;
    int query_start = thread_param.query_start;
    int query_end = thread_param.query_end;
    int n_query_data_thread = query_end - query_start;
    const float* query_data_ptr = query_data.data() + query_start * dim_query_data;
    const int* ground_truth_ptr = ground_truth.data() + query_start * dim_ground_truth;
    
    // --- Lambda: Run a timed benchmark in batches and compute recall after the loop ---
    auto run_ef_benchmark = [&](int ef, size_t duration_sec) {
        int batch_size = 2000;
        int top_k = 1;
        int branching_k = 8;
        int queries_executed = 0;
        double total_compute_time = 0.0;
        double total_network_latency = 0.0;
        double total_meta_search_time = 0.0;
        double total_batch_time = 0.0;
        
        // Vectors to accumulate results for recall computation
        std::vector<int> all_retrieved;
        std::vector<int> all_ground_truth;
        
        // Allocate arrays for batch processing
        float* batch_meta_distances = new float[branching_k * batch_size];
        dhnsw_idx_t* batch_meta_labels = new dhnsw_idx_t[branching_k * batch_size];
        dhnsw_idx_t* batch_sub_hnsw_tags = new dhnsw_idx_t[top_k * batch_size];
        dhnsw_idx_t* batch_labels = new dhnsw_idx_t[top_k * batch_size];
        float* batch_distances = new float[top_k * batch_size];
        dhnsw_idx_t* batch_original_index = new dhnsw_idx_t[top_k * batch_size];
        
        auto bench_start = high_resolution_clock::now();
        int query_index = 0;  // index within thread's query range
        
        while (duration_cast<seconds>(high_resolution_clock::now() - bench_start).count() < (long)duration_sec) {
            int current_batch_size = std::min(batch_size, n_query_data_thread - (query_index % n_query_data_thread));
            if (current_batch_size <= 0) {
                current_batch_size = batch_size;
            }
            const float* batch_query_data_ptr = query_data_ptr + ((query_index % n_query_data_thread) * dim_query_data);
            
            // --- Meta-search for the batch ---
            std::vector<int> sub_hnsw_tosearch_batch;
            //dhnsw without db
            auto start = high_resolution_clock::now();
            local_hnsw.hnsw.efSearch = ef;
            local_hnsw.search(current_batch_size, batch_query_data_ptr,top_k,batch_distances,batch_labels);
            auto end = high_resolution_clock::now();

            double duration = duration_cast<microseconds>(end - start).count();
            total_compute_time += duration;
            total_batch_time += duration;  // Add this line to properly track total batch time
            
            // --- Copy search results to batch_original_index for recall calculation ---
            for (int i = 0; i < current_batch_size; i++) {
                int pos = i * top_k;
                batch_original_index[pos] = batch_labels[pos];  // Copy the search results
            }
            
            // --- Accumulate results for recall (store each retrieved and corresponding ground truth) ---
            for (int i = 0; i < current_batch_size; i++) {
                int pos = i * top_k;
                int gt = *(ground_truth_ptr + ((query_index + i) % n_query_data_thread * dim_ground_truth));
                int retrieved = batch_original_index[pos];
                all_ground_truth.push_back(gt);
                all_retrieved.push_back(retrieved);
            }
            
            queries_executed += current_batch_size;
            query_index += current_batch_size;
        }
        
        // --- Calculate recall after processing all batches ---
        int total_correct = 0;
        for (size_t i = 0; i < all_retrieved.size(); i++) {
            if (all_retrieved[i] == all_ground_truth[i]) {
                total_correct++;
            }
        }
        float recall = (all_retrieved.size() > 0) ? static_cast<float>(total_correct) / all_retrieved.size() : 0.0f;
        double avg_total_latency = (queries_executed > 0) ? total_batch_time / queries_executed : 0.0;
        double avg_meta = (queries_executed > 0) ? total_meta_search_time / queries_executed : 0.0;
        double avg_compute = (queries_executed > 0) ? total_compute_time / queries_executed : 0.0;
        double avg_network = (queries_executed > 0) ? total_network_latency / queries_executed : 0.0;
        double throughput = (avg_total_latency > 0) ? queries_executed / (avg_total_latency * 1e-6) : 0.0;
        std::cout << "Thread " << thread_id << " EF " << ef << " benchmark:" << std::endl;
        std::cout << "  Queries executed: " << queries_executed << std::endl;
        std::cout << "  Avg total latency: " << avg_total_latency << " us" << std::endl;
        std::cout << "  Avg meta search latency: " << avg_meta << " us" << std::endl;
        std::cout << "  Avg compute time: " << avg_compute << " us" << std::endl;
        std::cout << "  Avg network latency: " << avg_network << " us" << std::endl;
        std::cout << "  Recall: " << recall << std::endl;
        std::cout << "  Throughput: " << throughput << " QPS" << std::endl;
        // Save results into thread parameters
        thread_param.per_ef_latencies.push_back(avg_total_latency);
        thread_param.per_ef_recalls.push_back(recall);
        thread_param.per_ef_network_latencies.push_back(avg_network);
        thread_param.per_ef_compute_times.push_back(avg_compute);
        thread_param.per_ef_duration_meta_search.push_back(avg_meta);
        thread_param.per_ef_throughput.push_back(throughput);
        delete[] batch_meta_distances;
        delete[] batch_meta_labels;
        delete[] batch_sub_hnsw_tags;
        delete[] batch_labels;
        delete[] batch_distances;
        delete[] batch_original_index;
    };
    
    // --- Run benchmark for each EF value ---
    for (int ef : ef_search_values) {
        std::cout << "=== Thread " << thread_id << " testing EF = " << ef 
                  << " for " << FLAGS_benchmark_duration << " seconds ===" << std::endl;
        run_ef_benchmark(ef, FLAGS_benchmark_duration);
    }
    
    // Don't delete the shared global index pointer
    // delete local_hnsw_ptr;  // Remove this line
    
    std::cout << "Thread " << thread_id << " returns" << std::endl;
    return nullptr;
}