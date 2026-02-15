// client_multithreads.cc

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
#include <numeric> // For std::accumulate
#include <random>  // For random number generation
#include "../dhnsw/DistributedHnsw.h"
#include "../util/read_dataset.h"
#include <omp.h>
#include <sched.h>  // For CPU pinning(core for each thread is fixed)

typedef int64_t dhnsw_idx_t;

DEFINE_string(server_address, "130.127.134.38:50051", "Address of the gRPC server.");
DEFINE_string(rdma_server_address, "192.168.1.2:8888", "Address of the RDMA server.");
DEFINE_int32(use_nic_idx, 3, "Which NIC to create QP");
// 0 for r320 3 for r650
DEFINE_int32(reg_nic_name, 73, "The name to register an opened NIC at rctrl.");
DEFINE_int32(reg_mem_name, 73, "The name to register an MR at rctrl.");
DEFINE_string(query_data_path, "../datasets/sift/sift_query.fvecs", "Path to the query data.");
DEFINE_string(ground_truth_path, "../datasets/sift/sift_groundtruth.ivecs", "Path to the ground truth data.");
DEFINE_double(search_ratio, 0.7, "Ratio of search operations (between 0 and 1). The rest are insert operations.");

// DEFINE_string(query_data_path, "../datasets/gist/gist_query.fvecs", "Path to the query data.");
// DEFINE_string(ground_truth_path, "../datasets/gist/gist_groundtruth.ivecs", "Path to the ground truth data.");

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using dhnsw::DhnswService;
using dhnsw::Empty;
using dhnsw::MetaHnswResponse;
using dhnsw::OffsetResponse;
using dhnsw::MappingResponse;
using dhnsw::MappingEntry;
using dhnsw::PartResponse;

using namespace rdmaio;
using namespace rdmaio::rmem;
using namespace rdmaio::qp;

using namespace std::chrono;

// Global variables
std::vector<float> query_data;
std::vector<int> ground_truth;
int dim_query_data;
int n_query_data;
int dim_ground_truth;

struct thread_param_t {
    int thread_id;
    int omp_threads_per_worker;
    int core_start;  // Starting core ID for this thread , range will be [core_start, core_start + omp_threads_per_worker - 1]
    size_t throughput;
    double latency;
    // Separate metrics for search and insert operations
    size_t search_throughput;
    double search_latency;
    size_t search_count;
    size_t insert_throughput;
    double insert_latency;
    size_t insert_count;
    double search_ratio;  // Ratio of search operations (between 0 and 1)
    int query_start;
    int query_end;
    int n_queries_processed;
    Arc<RMem> local_mem;
    Arc<RegHandler> local_mr;
    Arc<RNic> nic;
    std::shared_ptr<RC> qp;
    rmem::RegAttr remote_attr;
    uint64_t key; // Authentication key
};
void* client_worker(void* param);

void bind_thread_to_cores(int thread_id, int core_start, int cores_per_thread) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    
    // Set affinity to assigned cores
    for (int i = 0; i < cores_per_thread; i++) {
        int core_id = core_start + i;
        CPU_SET(core_id, &cpuset);
    }
    
    // Apply the CPU affinity mask to the current thread
    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error: unable to set thread affinity for thread " << thread_id 
                  << " to cores " << core_start << "-" << (core_start + cores_per_thread - 1) 
                  << ", error code: " << rc << std::endl;
    } else {
        std::cout << "Thread " << thread_id << " bound to cores " 
                  << core_start << "-" << (core_start + cores_per_thread - 1) << std::endl;
    }
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    // Read data
    std::string query_data_path = FLAGS_query_data_path;
    std::string ground_truth_path = FLAGS_ground_truth_path;

    std::vector<float> query_data_tmp;
    std::vector<int> ground_truth_tmp;
    int n_query_data_tmp;
    int n_ground_truth_tmp;
    query_data_tmp = read_fvecs(query_data_path, dim_query_data, n_query_data_tmp);
    ground_truth_tmp = read_ivecs(ground_truth_path, dim_ground_truth, n_ground_truth_tmp);

    // Sample 1/3 the query data
    for(int i = 0; i < n_query_data_tmp; i++) {
        if(i % 3 == 0){ // sample 1/3 of the query data (remember to change)
            query_data.insert(query_data.end(),
                query_data_tmp.begin() + i * dim_query_data,
                query_data_tmp.begin() + (i + 1) * dim_query_data);

            ground_truth.insert(ground_truth.end(),
                ground_truth_tmp.begin() + i * dim_ground_truth,
                ground_truth_tmp.begin() + (i + 1) * dim_ground_truth);
        }
    }
    // remember to change
    n_query_data = 800;
    
    // ground_truth cut off to maintain 1000 queries
    ground_truth.resize(n_query_data * dim_ground_truth);

    // Define the maximum number of threads
    const int max_threads = 8;
    int omp_threads_per_worker = 18;
    int total_cores = max_threads * omp_threads_per_worker; 
  
    // Initialize RDMA resources in the main function
    // Create a ConnectManager and connect to the server's controller
    ConnectManager cm(FLAGS_rdma_server_address);
    if (cm.wait_ready(1000000, 2) == IOCode::Timeout) {
        RDMA_LOG(4) << "cm connect to server timeout";
        return -1;
    }

    // Fetch the server's MR info
    auto fetch_res = cm.fetch_remote_mr(FLAGS_reg_mem_name);
    RDMA_ASSERT(fetch_res == IOCode::Ok) << std::get<0>(fetch_res.desc);
    rmem::RegAttr remote_attr = std::get<1>(fetch_res.desc);

    // Initialize NIC and local memory for all possible threads
    std::vector<Arc<RNic>> nics(max_threads);
    std::vector<std::shared_ptr<RC>> qps(max_threads);
    std::vector<Arc<RMem>> local_mems(max_threads);
    std::vector<Arc<RegHandler>> local_mrs(max_threads);
    std::vector<uint64_t> keys(max_threads);

    for (int i = 0; i < max_threads; ++i) {
        // Initialize per-thread RDMA resources
        nics[i] = RNic::create(RNicInfo::query_dev_names().at(FLAGS_use_nic_idx)).value();
        qps[i] = RC::create(nics[i], QPConfig()).value();

        // Use a unique name for the QP
        std::string qp_name = "client1-" + std::to_string(i);
        auto qp_res = cm.cc_rc(qp_name, qps[i], FLAGS_reg_nic_name, QPConfig());
        RDMA_ASSERT(qp_res == IOCode::Ok) << std::get<0>(qp_res.desc);
        keys[i] = std::get<1>(qp_res.desc);
        RDMA_LOG(4) << "Thread " << i << " fetch QP authentical key: " << keys[i];

        // Register local memory
        size_t fixed_size = 1024 * 1024 * 1024; // Adjust size as needed
        local_mems[i] = Arc<RMem>(new RMem(fixed_size));
        local_mrs[i] = RegHandler::create(local_mems[i], nics[i]).value();

        RDMA_LOG(4) << "Thread " << i << " RDMA resources initialized";
    }

    // Run tests with varying number of threads
    std::vector<int> thread_counts = {1, 2, 4, 8}; // Testing with 1, 2, 4, and 8 threads
    std::vector<double> avg_latencies;
    std::vector<double> throughputs;

    for (int num_threads : thread_counts) {
        std::cout << "Testing with " << num_threads << " threads" << std::endl;
        
        std::vector<pthread_t> threads(num_threads);
        std::vector<thread_param_t> thread_params(num_threads);

        // Divide queries evenly among threads
        int queries_per_thread = n_query_data / num_threads;
        
        for (int i = 0; i < num_threads; ++i) {
            thread_params[i].thread_id = i;
            thread_params[i].throughput = 0;
            thread_params[i].latency = 0;
            thread_params[i].search_throughput = 0;
            thread_params[i].search_latency = 0;
            thread_params[i].search_count = 0;
            thread_params[i].insert_throughput = 0;
            thread_params[i].insert_latency = 0;
            thread_params[i].insert_count = 0;
            thread_params[i].search_ratio = FLAGS_search_ratio;
            thread_params[i].omp_threads_per_worker = omp_threads_per_worker;
            thread_params[i].core_start = i * omp_threads_per_worker;
            // Assign query ranges
            thread_params[i].query_start = i * queries_per_thread;
            thread_params[i].query_end = (i == num_threads - 1) ? n_query_data : (i + 1) * queries_per_thread;
            thread_params[i].n_queries_processed = thread_params[i].query_end - thread_params[i].query_start;

            // Pass RDMA resources to threads
            thread_params[i].nic = nics[i];
            thread_params[i].qp = qps[i];
            thread_params[i].local_mem = local_mems[i];
            thread_params[i].local_mr = local_mrs[i];
            thread_params[i].remote_attr = remote_attr;
            thread_params[i].key = keys[i];

            // Create the thread
            int ret = pthread_create(&threads[i], nullptr, client_worker, (void*)&thread_params[i]);
            if (ret != 0) {
                std::cerr << "Error: unable to create thread," << ret << std::endl;
                exit(-1);
            }
        }

        // Wait for threads to finish
        for (int i = 0; i < num_threads; ++i) {
            void* status;
            int rc = pthread_join(threads[i], &status);
            if (rc) {
                std::cerr << "Error: unable to join, " << rc << std::endl;
                exit(-1);
            }
        }

        // Calculate average latency and total throughput
        double total_latency = 0.0;
        double total_search_latency = 0.0;
        double total_insert_latency = 0.0;
        size_t total_queries_processed = 0;
        size_t total_search_count = 0;
        size_t total_insert_count = 0;

        for (int i = 0; i < num_threads; ++i) {
            total_latency += thread_params[i].latency;
            total_search_latency += thread_params[i].search_latency;
            total_insert_latency += thread_params[i].insert_latency;
            total_queries_processed += thread_params[i].n_queries_processed;
            total_search_count += thread_params[i].search_count;
            total_insert_count += thread_params[i].insert_count;
        }

        double avg_latency = total_latency / num_threads;
        double throughput = static_cast<double>(total_queries_processed) / (avg_latency / 1000000.0); // queries per second
        
        double avg_search_latency = total_search_count > 0 ? total_search_latency / total_search_count : 0;
        double avg_insert_latency = total_insert_count > 0 ? total_insert_latency / total_insert_count : 0;
        double search_throughput = total_search_count > 0 ? static_cast<double>(total_search_count) / (total_search_latency / 1000000.0) : 0;
        double insert_throughput = total_insert_count > 0 ? static_cast<double>(total_insert_count) / (total_insert_latency / 1000000.0) : 0;

        avg_latencies.push_back(avg_latency);
        throughputs.push_back(throughput);

        std::cout << "Threads: " << num_threads << 
                  ", Avg Latency (us): " << avg_latency << 
                  ", Throughput (q/s): " << throughput << std::endl;
        std::cout << "  Search Ratio: " << FLAGS_search_ratio << 
                  ", Search Latency (us): " << avg_search_latency <<
                  ", Search Throughput (q/s): " << search_throughput << std::endl;
        std::cout << "  Insert Ratio: " << (1 - FLAGS_search_ratio) << 
                  ", Insert Latency (us): " << avg_insert_latency <<
                  ", Insert Throughput (q/s): " << insert_throughput << std::endl;
    }

    // Write results to output file
    std::ofstream outfile("../benchs/dhnsw_withoutdb/sift1M@10benchmark_mixed_throughput_latency.txt");
    outfile << "search_ratio\tthreads\tlatency(us)\tthroughput(q/s)" << std::endl;
    for (size_t i = 0; i < thread_counts.size(); ++i) {
        outfile << FLAGS_search_ratio << "\t"
                << thread_counts[i] << "\t" 
                << avg_latencies[i] << "\t" 
                << throughputs[i] << std::endl;
    }
    outfile.close();

    // Write more detailed results to a separate file
    std::string detailed_filename = "../benchs/dhnsw_withoutdb/sift1M@10benchmark_mixed_detailed_" + 
                                  std::to_string(FLAGS_search_ratio) + ".txt";
    std::ofstream detailed_outfile(detailed_filename);
    detailed_outfile << "threads\tsearch_latency(us)\tsearch_throughput(q/s)\tinsert_latency(us)\tinsert_throughput(q/s)" << std::endl;
    for (int num_threads : thread_counts) {
        // These values were already calculated and printed to console for each test
        detailed_outfile << num_threads << "\t"
                        << "see_console_output" << "\t"
                        << "see_console_output" << "\t"
                        << "see_console_output" << "\t"
                        << "see_console_output" << std::endl;
    }
    detailed_outfile.close();

    // Clean up RDMA resources 
    for (int i = 0; i < max_threads; ++i) {
        std::string qp_name = "client1-" + std::to_string(i);
        auto del_res = cm.delete_remote_rc(qp_name, keys[i]);
        RDMA_ASSERT(del_res == IOCode::Ok)
            << "Thread " << i << " delete remote QP error: " << del_res.desc;
        qps[i].reset();
    }

    return 0;
}

void* client_worker(void* param) {
    thread_param_t& thread_param = *(thread_param_t*)param;
    int thread_id = thread_param.thread_id;
    bind_thread_to_cores(thread_id, thread_param.core_start, thread_param.omp_threads_per_worker);
   
    omp_set_num_threads(thread_param.omp_threads_per_worker);

    // Initialize random number generator for deciding between search and insert
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    try {
        // RDMA resources are already initialized and passed via thread_param

        // Initialize DhnswClient
        DhnswClient dhnsw_client(grpc::CreateChannel(FLAGS_server_address, grpc::InsecureChannelCredentials())); 

        // Initialize LocalHnsw
        int dim = 128;
        int num_sub_hnsw = 80;
        int meta_hnsw_neighbors = 32;
        int sub_hnsw_neighbors = 48;
        LocalHnsw local_hnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, &dhnsw_client);
        local_hnsw.init();

        // Set RDMA resources
        local_hnsw.set_rdma_qp(thread_param.qp.get(), thread_param.remote_attr, thread_param.local_mr);
        local_hnsw.set_remote_attr(thread_param.remote_attr);
        local_hnsw.set_local_mr(thread_param.local_mr, thread_param.local_mem);

        // Process assigned queries
        int query_start = thread_param.query_start;
        int query_end = thread_param.query_end;
        int n_query_data_thread = query_end - query_start;
        thread_param.n_queries_processed = n_query_data_thread;

        // Prepare data pointers
        const float* query_data_ptr = query_data.data() + query_start * dim_query_data;
        const int* ground_truth_ptr = ground_truth.data() + query_start * dim_ground_truth;

        int branching_k = 5;
        int top_k = 1;
        int ef = 128;

        // Track metrics
        double total_duration = 0.0;
        double search_duration = 0.0;
        double insert_duration = 0.0;
        size_t search_count = 0;
        size_t insert_count = 0;

        // Initialize data structures for search
        std::vector<int> sub_hnsw_tosearch;
        std::vector<std::vector<dhnsw_idx_t>> mapping = local_hnsw.get_local_mapping(); // For easy testing
        
        local_hnsw.set_meta_ef_search(ef);
        std::vector<int> local_sub_hnsw_tag;
        local_hnsw.set_local_sub_hnsw_tag(local_sub_hnsw_tag);

        // Allocate memory for search results
        float* meta_distances = new float[branching_k * n_query_data_thread];
        dhnsw_idx_t* meta_labels = new dhnsw_idx_t[branching_k * n_query_data_thread];
        dhnsw_idx_t* sub_hnsw_tags = new dhnsw_idx_t[top_k * n_query_data_thread];
        dhnsw_idx_t* labels = new dhnsw_idx_t[top_k * n_query_data_thread];
        float* distances = new float[top_k * n_query_data_thread];
        
        // For each query, randomly decide whether to search or insert
        for (int i = 0; i < n_query_data_thread; i++) {
            bool do_search = (dis(gen) < thread_param.search_ratio);
            
            if (do_search) {
                // Perform search operation
                auto start = high_resolution_clock::now();
                
                // Get the query vector for this iteration
                const float* query_vec = query_data_ptr + i * dim_query_data;
                
                // Clear sub_hnsw_tosearch for this query
                sub_hnsw_tosearch.clear();
                
                // Perform meta-search first
                local_hnsw.meta_search(1, query_vec, branching_k, 
                                      meta_distances + i * branching_k, 
                                      meta_labels + i * branching_k, 
                                      sub_hnsw_tosearch);
                
                // Then perform sub-search
                local_hnsw.sub_search_each(1, query_vec, branching_k, top_k, 
                                         distances + i * top_k, 
                                         labels + i * top_k, 
                                         sub_hnsw_tosearch, 
                                         sub_hnsw_tags + i * top_k, 
                                         ef, RDMA_DOORBELL);
                
                auto stop = high_resolution_clock::now();
                auto duration = duration_cast<microseconds>(stop - start).count();
                
                // Update metrics
                search_duration += duration;
                search_count++;
                total_duration += duration;
            } else {
                // Perform insert operation
                auto start = high_resolution_clock::now();
                
                // Get the query vector for this iteration and convert to vector
                const float* query_vec = query_data_ptr + i * dim_query_data;
                std::vector<float> insert_data(query_vec, query_vec + dim);
                
                // Clear sub_hnsw_tosearch for this insert operation
                std::vector<int> sub_hnsw_toinsert;
                
                // Perform insert operation
                local_hnsw.insert_rdma(1, insert_data, RDMA_DOORBELL, sub_hnsw_toinsert);
                
                auto stop = high_resolution_clock::now();
                auto duration = duration_cast<microseconds>(stop - start).count();
                
                // Update metrics
                insert_duration += duration;
                insert_count++;
                total_duration += duration;
            }
        }
        
        // Update thread parameter with metrics
        thread_param.search_count = search_count;
        thread_param.insert_count = insert_count;
        thread_param.search_latency = search_count > 0 ? search_duration / search_count : 0;
        thread_param.insert_latency = insert_count > 0 ? insert_duration / insert_count : 0;
        thread_param.latency = total_duration / n_query_data_thread;
        
        // Clean up
        delete[] labels;
        delete[] distances;
        delete[] sub_hnsw_tags;
        delete[] meta_distances;
        delete[] meta_labels;

        RDMA_LOG(4) << "Thread " << thread_id 
                   << " processed " << n_query_data_thread << " queries in " << total_duration << " us"
                   << " (Search: " << search_count << ", Insert: " << insert_count << ")";

    } catch (const std::exception& e) {
        std::cerr << "Thread " << thread_id << " encountered an exception: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "Thread " << thread_id << " encountered an unknown exception." << std::endl;
        return nullptr;
    }
    return nullptr;
}