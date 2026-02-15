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
#include "../dhnsw/DistributedHnsw.h"
#include "../util/read_dataset.h"
#include <omp.h>

typedef int64_t dhnsw_idx_t;

DEFINE_string(server_address, "130.127.134.38:50051", "Address of the gRPC server.");
DEFINE_string(rdma_server_address, "192.168.1.2:8888", "Address of the RDMA server.");
DEFINE_int32(use_nic_idx, 3, "Which NIC to create QP");
// 0 for r320 3 for r650
DEFINE_int32(reg_nic_name, 73, "The name to register an opened NIC at rctrl.");
DEFINE_int32(reg_mem_name, 73, "The name to register an MR at rctrl.");
// SIFT1B dataset paths
DEFINE_string(query_data_path, "../datasets/sift1b/sift_query.fvecs", "Path to the SIFT1B query data.");
DEFINE_string(ground_truth_path, "../datasets/sift1b/sift_groundtruth.ivecs", "Path to the SIFT1B ground truth data.");

// Previous dataset paths commented out for reference
// DEFINE_string(query_data_path, "../datasets/sift/sift_query.fvecs", "Path to the query data.");
// DEFINE_string(ground_truth_path, "../datasets/sift/sift_groundtruth.ivecs", "Path to the ground truth data.");
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
    int query_start;
    int query_end;
    std::vector<float> per_ef_recalls;
    std::vector<double> per_ef_latencies;
    Arc<RMem> local_mem;
    Arc<RegHandler> local_mr;
    Arc<RNic> nic;
    std::shared_ptr<RC> qp;
    rmem::RegAttr remote_attr;
    uint64_t key; // Authentication key
    std::vector<double> per_ef_compute_times;
    std::vector<double> per_ef_network_latencies;
    std::vector<double> per_ef_duration_meta_search;
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

    std::cout << "Loading SIFT1B query data from: " << query_data_path << std::endl;
    std::cout << "Loading ground truth data from: " << ground_truth_path << std::endl;

    std::vector<float> query_data_tmp;
    std::vector<int> ground_truth_tmp;
    int n_query_data_tmp;
    int n_ground_truth_tmp;
    query_data_tmp = read_fvecs(query_data_path, dim_query_data, n_query_data_tmp);
    ground_truth_tmp = read_ivecs(ground_truth_path, dim_ground_truth, n_ground_truth_tmp);

    std::cout << "Query data dimension: " << dim_query_data << ", count: " << n_query_data_tmp << std::endl;
    std::cout << "Ground truth dimension: " << dim_ground_truth << ", count: " << n_ground_truth_tmp << std::endl;

    // For SIFT1B, we'll use a subset of queries for testing
    // We can use different sampling strategies based on needs
    n_query_data = 10000; // Adjust based on your testing needs

    // Sample queries evenly from the dataset
    int sample_step = std::max(1, n_query_data_tmp / n_query_data);
    for(int i = 0; i < n_query_data_tmp; i += sample_step) {
        if(query_data.size() / dim_query_data >= n_query_data) break;
        
        query_data.insert(query_data.end(),
            query_data_tmp.begin() + i * dim_query_data,
            query_data_tmp.begin() + (i + 1) * dim_query_data);

        ground_truth.insert(ground_truth.end(),
            ground_truth_tmp.begin() + i * dim_ground_truth,
            ground_truth_tmp.begin() + (i + 1) * dim_ground_truth);
    }

    // Adjust n_query_data in case we didn't get exactly the number we wanted
    n_query_data = query_data.size() / dim_query_data;
    
    // Ensure ground_truth matches n_query_data
    ground_truth.resize(n_query_data * dim_ground_truth);

    std::cout << "Using " << n_query_data << " queries for evaluation" << std::endl;

    int num_threads = 8;
    int omp_threads_per_worker = 18;
    int total_cores = num_threads * omp_threads_per_worker; 

    std::vector<pthread_t> threads(num_threads);
    std::vector<thread_param_t> thread_params(num_threads);

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

    // Initialize NIC and local memory
    std::vector<Arc<RNic>> nics(num_threads);
    std::vector<std::shared_ptr<RC>> qps(num_threads);
    std::vector<Arc<RMem>> local_mems(num_threads);
    std::vector<Arc<RegHandler>> local_mrs(num_threads);
    std::vector<uint64_t> keys(num_threads);

    for (int i = 0; i < num_threads; ++i) {
        // Initialize per-thread RDMA resources
        nics[i] = RNic::create(RNicInfo::query_dev_names().at(FLAGS_use_nic_idx)).value();
        qps[i] = RC::create(nics[i], QPConfig()).value();

        // Use a unique name for the QP
        std::string qp_name = "client_sift1b-" + std::to_string(i);
        auto qp_res = cm.cc_rc(qp_name, qps[i], FLAGS_reg_nic_name, QPConfig());
        RDMA_ASSERT(qp_res == IOCode::Ok) << std::get<0>(qp_res.desc);
        keys[i] = std::get<1>(qp_res.desc);
        RDMA_LOG(4) << "Thread " << i << " fetch QP authentical key: " << keys[i];

        // Increase local memory allocation for SIFT1B
        size_t fixed_size = 2000 * 1024 * 1024; // 2GB, adjust as needed  
        local_mems[i] = Arc<RMem>(new RMem(fixed_size));
        local_mrs[i] = RegHandler::create(local_mems[i], nics[i]).value();

        RDMA_LOG(4) << "Thread " << i << " RDMA resources initialized";
    }

    // Split queries among threads
    int queries_per_thread = n_query_data / num_threads;
    
    for (int i = 0; i < num_threads; ++i) {
        thread_params[i].thread_id = i;
        thread_params[i].throughput = 0;
        thread_params[i].latency = 0;
        thread_params[i].omp_threads_per_worker = omp_threads_per_worker;
        thread_params[i].core_start = i * omp_threads_per_worker;
        // Assign per-thread query ranges
        thread_params[i].query_start = i * queries_per_thread;
        thread_params[i].query_end = (i == num_threads - 1) ? n_query_data : (i + 1) * queries_per_thread;
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
    // Aggregate results per ef value
    // Adjust ef values for larger dataset if needed
    std::vector<int> ef_search_values = {1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512};
    std::vector<int> ef_values;
    std::vector<float> avg_recalls;
    std::vector<double> avg_latencies;
    std::vector<double> avg_network_latency;    
    std::vector<double> avg_duration_meta_search;
    std::vector<double> avg_compute_time;
    for (size_t ef_idx = 0; ef_idx < ef_search_values.size(); ++ef_idx) {
        int ef = ef_search_values[ef_idx];
        ef_values.push_back(ef);
        float sum_recalls = 0.0f;
        double sum_latency = 0.0;
        double sum_network_latency = 0.0;
        double sum_duration_meta_search = 0.0;
        double sum_compute_time = 0.0;
        
        std::cout << "EF: " << ef << std::endl;
        for (int t = 0; t < num_threads; ++t) {
            if (ef_idx < thread_params[t].per_ef_recalls.size()) {
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
            }
        }
        avg_recalls.push_back(sum_recalls / num_threads);
        avg_latencies.push_back(sum_latency / num_threads);
        avg_network_latency.push_back(sum_network_latency / num_threads);
        avg_duration_meta_search.push_back(sum_duration_meta_search / num_threads);
        avg_compute_time.push_back(sum_compute_time / num_threads);
    }

    // Write results to output file
    std::ofstream outfile("../benchs/hot_storage/search/dhnsw_withoutdb/sift1b_benchmark_results.txt");
    outfile << "latency(us)\trecall" << std::endl;
    for (size_t i = 0; i < ef_values.size(); ++i) {
        outfile << "[" <<  avg_latencies[i] << ", " << avg_recalls[i] <<"],"<< std::endl;
    }
    outfile.close();

    std::ofstream outfile2("../benchs/hot_storage/search/dhnsw_withoutdb/sift1b_benchmark_details.txt");
    outfile2 << "ef\tlatency(us)\trecall\tnetwork_latency(us)\tcompute_time(us)\tduration_meta_search(us)" << std::endl;
    for (size_t i = 0; i < ef_values.size(); ++i) {
        outfile2 << ef_values[i] << "\t" 
                 << avg_latencies[i] << "\t" 
                 << avg_recalls[i] << "\t" 
                 << avg_network_latency[i] << "\t" 
                 << avg_compute_time[i] << "\t" 
                 << avg_duration_meta_search[i] << std::endl;
    }
    outfile2.close();
    
    // Clean up RDMA resources 
    for (int i = 0; i < num_threads; ++i) {
        std::string qp_name = "client_sift1b-" + std::to_string(i);
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
    bind_thread_to_cores(thread_id, thread_param.core_start, thread_param.omp_threads_per_worker);
  
    omp_set_num_threads(thread_param.omp_threads_per_worker);
  
    // Adjust ef values for SIFT1B
    std::vector<int> ef_search_values = {1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512};
  
    thread_param.per_ef_recalls.resize(ef_search_values.size(), 0.0f);
    thread_param.per_ef_latencies.resize(ef_search_values.size(), 0.0);
    thread_param.per_ef_network_latencies.resize(ef_search_values.size(), 0.0);
    thread_param.per_ef_compute_times.resize(ef_search_values.size(), 0.0);
    thread_param.per_ef_duration_meta_search.resize(ef_search_values.size(), 0.0);

    try {
        // RDMA resources are already initialized and passed via thread_param

        // Initialize DhnswClient
        DhnswClient dhnsw_client(grpc::CreateChannel(FLAGS_server_address, grpc::InsecureChannelCredentials())); 

        // Initialize LocalHnsw - updated for SIFT1B
        int dim = 128; // SIFT dimension
        int num_sub_hnsw = 250;
        int meta_hnsw_neighbors = 32;
        int sub_hnsw_neighbors = 48;
        LocalHnsw local_hnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, &dhnsw_client);
        local_hnsw.init();
        std::cout << "Thread " << thread_id << " - LocalHnsw initialized with mapping size: " 
                  << local_hnsw.get_local_mapping().size() << std::endl;
        
        // Set RDMA resources
        local_hnsw.set_rdma_qp(thread_param.qp.get(), thread_param.remote_attr, thread_param.local_mr);
        local_hnsw.set_remote_attr(thread_param.remote_attr);
        local_hnsw.set_local_mr(thread_param.local_mr, thread_param.local_mem);

        // Process assigned queries
        int query_start = thread_param.query_start;
        int query_end = thread_param.query_end;
        int n_query_data_thread = query_end - query_start;
        
        // Prepare data pointers
        const float* query_data_ptr = query_data.data() + query_start * dim_query_data;
        const int* ground_truth_ptr = ground_truth.data() + query_start * dim_ground_truth;
        
        // For SIFT1B, we might want to increase branching_k
        int branching_k = 10; // Increased from 5
        int top_k = 1;

        std::vector<int> sub_hnsw_tosearch;
        std::vector<std::vector<dhnsw_idx_t>> mapping = local_hnsw.get_local_mapping(); // For easy testing
        
        for (size_t ef_idx = 0; ef_idx < ef_search_values.size(); ++ef_idx) {
            int ef = ef_search_values[ef_idx];
            std::vector<int> local_sub_hnsw_tag;
            local_hnsw.set_local_sub_hnsw_tag(local_sub_hnsw_tag);
            local_hnsw.set_meta_ef_search(ef);

            // Allocate memory for search results
            float* meta_distances = new float[branching_k * n_query_data_thread];
            dhnsw_idx_t* meta_labels = new dhnsw_idx_t[branching_k * n_query_data_thread];
            dhnsw_idx_t* sub_hnsw_tags = new dhnsw_idx_t[top_k * n_query_data_thread];
            dhnsw_idx_t* labels = new dhnsw_idx_t[top_k * n_query_data_thread];
            float* distances = new float[top_k * n_query_data_thread];
            
            std::cout << "Thread " << thread_id << " - Testing ef=" << ef 
                      << " for queries " << query_start << " to " << query_end << std::endl;
            
            auto start = high_resolution_clock::now();
            
            // Meta search
            auto start_meta_search = high_resolution_clock::now();
            local_hnsw.meta_search(n_query_data_thread, query_data_ptr, branching_k, 
                                  meta_distances, meta_labels, sub_hnsw_tosearch);
            auto stop_meta_search = high_resolution_clock::now();
            auto duration_meta_search = duration_cast<microseconds>(stop_meta_search - start_meta_search).count();
            
            // For SIFT1B, prefer RDMA_DOORBELL for better performance with large data
            std::pair<double, double> result = local_hnsw.sub_search_each(
                n_query_data_thread, query_data_ptr, branching_k, top_k, 
                distances, labels, sub_hnsw_tosearch, sub_hnsw_tags, ef, RDMA_DOORBELL);
            
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start).count();
            
            double total_compute_time = result.first;
            double total_network_latency = result.second;
            
            thread_param.per_ef_latencies[ef_idx] = duration/n_query_data_thread;
            thread_param.per_ef_network_latencies[ef_idx] = result.second/n_query_data_thread;
            thread_param.per_ef_compute_times[ef_idx] = result.first/n_query_data_thread;
            thread_param.per_ef_duration_meta_search[ef_idx] = duration_meta_search/n_query_data_thread;
            
            dhnsw_idx_t *original_index = new dhnsw_idx_t[top_k * n_query_data_thread];
            for(int i = 0; i < n_query_data_thread; i++) {
                for(int j = 0; j < top_k; j++) {
                    if (sub_hnsw_tags[i * top_k + j] < mapping.size() && 
                        labels[i * top_k + j] < mapping[sub_hnsw_tags[i * top_k + j]].size()) {
                        original_index[i * top_k + j] = mapping[sub_hnsw_tags[i * top_k + j]][labels[i * top_k + j]];
                    } else {
                        std::cerr << "Thread " << thread_id << " - Index out of bounds: sub_hnsw_tag=" 
                                  << sub_hnsw_tags[i * top_k + j] << ", label=" << labels[i * top_k + j] 
                                  << ", mapping size=" << mapping.size() << std::endl;
                        original_index[i * top_k + j] = -1; // Invalid index
                    }
                }
            }
            
            int correct = 0;
            float recall = 0.0f;
            for(int i = 0; i < n_query_data_thread; i++) {
                std::unordered_set<int> ground_truth_set(
                    ground_truth_ptr + i * dim_ground_truth,
                    ground_truth_ptr + i * dim_ground_truth + top_k
                );

                for(int j = 0; j < top_k; j++) {
                    int retrieved_idx = original_index[i * top_k + j];
                    if (retrieved_idx >= 0 && ground_truth_set.find(retrieved_idx) != ground_truth_set.end()) {
                        correct++;
                    }
                }
            }

            recall = static_cast<float>(correct) / (n_query_data_thread * top_k); 
            thread_param.per_ef_recalls[ef_idx] = recall;
            
            std::cout << "Thread " << thread_id << " - ef=" << ef 
                      << " completed with recall=" << recall 
                      << ", avg latency=" << thread_param.per_ef_latencies[ef_idx] << "us" << std::endl;
    
            // Clean up
            delete[] labels;
            delete[] distances;
            delete[] sub_hnsw_tags;
            delete[] meta_distances;
            delete[] meta_labels;
            delete[] original_index;
        }

        // Collect per-thread throughput and latency
        thread_param.throughput = n_query_data_thread;
        // Average latency across all ef values
        thread_param.latency = std::accumulate(
            thread_param.per_ef_latencies.begin(), 
            thread_param.per_ef_latencies.end(), 0.0) / thread_param.per_ef_latencies.size();

        RDMA_LOG(4) << "Thread " << thread_id << " completed all tests";

    } catch (const std::exception& e) {
        std::cerr << "Thread " << thread_id << " encountered an exception: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "Thread " << thread_id << " encountered an unknown exception." << std::endl;
        return nullptr;
    }
    return nullptr;
}