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
#include <unistd.h>
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

DEFINE_string(client_id, "", "Unique client identifier for multi-node experiments.");

// DEFINE_string(query_data_path, "../datasets/gist/gist_query.fvecs", "Path to the query data.");
// DEFINE_string(ground_truth_path, "../datasets/gist/gist_groundtruth.ivecs", "Path to the ground truth data.");

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
    query_data.clear();
    ground_truth.clear();
    for (int i = 0; i < n_query_data_tmp; i++) {
        if (i % 3 == 0) { // sample 1/3 of the queries
            query_data.insert(query_data.end(),
                              query_data_tmp.begin() + i * dim_query_data,
                              query_data_tmp.begin() + (i + 1) * dim_query_data);
            ground_truth.insert(ground_truth.end(),
                                ground_truth_tmp.begin() + i * dim_ground_truth,
                                ground_truth_tmp.begin() + (i + 1) * dim_ground_truth);
        }
    }
    // Here, we determine the actual number of sampled queries.
    n_query_data = query_data.size() / dim_query_data;
    if (n_query_data == 0) {
        std::cerr << "Error: No queries were sampled." << std::endl;
        return -1;
    }
    std::cout << "Total sampled queries: " << n_query_data << std::endl;

    // ground_truth is trimmed to match query_data size.
    ground_truth.resize(n_query_data * dim_ground_truth);

    // Set test parameters.
    const int max_threads = 8;
    int omp_threads_per_worker = 18;
    int total_cores = max_threads * omp_threads_per_worker;

    // --- Generate unique client ID for multi-node QP naming ---
    std::string unique_id = FLAGS_client_id;
    if (unique_id.empty()) {
        auto ts = std::chrono::system_clock::now().time_since_epoch().count();
        unique_id = std::to_string(getpid()) + "-" + std::to_string(ts);
    }
    std::cout << "Client unique ID: " << unique_id << std::endl;

    // Create a ConnectManager and connect to the RDMA server's controller.
    ConnectManager cm(FLAGS_rdma_server_address);
    if (cm.wait_ready(1000000, 20) == IOCode::Timeout) {
        RDMA_LOG(4) << "cm connect to server timeout";
        return -1;
    }
    // Fetch the remote MR info once (this is global and reused).
    auto fetch_res = cm.fetch_remote_mr(FLAGS_reg_mem_name);
    RDMA_ASSERT(fetch_res == IOCode::Ok) << std::get<0>(fetch_res.desc);
    rmem::RegAttr remote_attr = std::get<1>(fetch_res.desc);

    // Run tests with varying number of threads.
    std::vector<int> thread_counts = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<double> avg_latencies;
    std::vector<double> throughputs;

    for (int num_threads : thread_counts) {
        std::cout << "Testing with " << num_threads << " threads" << std::endl;
        
        // Allocate per-thread RDMA resources for this test.
        std::vector<Arc<RNic>> nics(num_threads);
        std::vector<std::shared_ptr<RC>> qps(num_threads);
        std::vector<Arc<RMem>> local_mems(num_threads);
        std::vector<Arc<RegHandler>> local_mrs(num_threads);
        std::vector<uint64_t> keys(num_threads);
        
        for (int i = 0; i < num_threads; ++i) {
            // Initialize per-thread RDMA resources.
            nics[i] = RNic::create(RNicInfo::query_dev_names().at(FLAGS_use_nic_idx)).value();
            qps[i] = RC::create(nics[i], QPConfig()).value();
            
            // Use a unique name for the QP.
            std::string qp_name = unique_id + "-" + std::to_string(num_threads) + "-" + std::to_string(i);
            auto qp_res = cm.cc_rc(qp_name, qps[i], FLAGS_reg_nic_name, QPConfig());
            RDMA_ASSERT(qp_res == IOCode::Ok) << std::get<0>(qp_res.desc);
            keys[i] = std::get<1>(qp_res.desc);
            RDMA_LOG(4) << "Thread " << i << " fetch QP authentical key: " << keys[i];
            
            // Register local memory.
            size_t fixed_size = 1024 * 1024 * 1024; // Adjust size as needed.
            local_mems[i] = Arc<RMem>(new RMem(fixed_size));
            local_mrs[i] = RegHandler::create(local_mems[i], nics[i]).value();
            RDMA_LOG(4) << "Thread " << i << " RDMA resources initialized";
        }

        // Create threads.
        std::vector<pthread_t> threads(num_threads);
        std::vector<thread_param_t> thread_params(num_threads);
        
        // Divide queries evenly among threads.
        int queries_per_thread = n_query_data / num_threads;
        for (int i = 0; i < num_threads; ++i) {
            thread_params[i].thread_id = i;
            thread_params[i].throughput = 0;
            thread_params[i].latency = 0;
            thread_params[i].omp_threads_per_worker = omp_threads_per_worker;
            thread_params[i].core_start = i * omp_threads_per_worker;
            // Assign query ranges.
            thread_params[i].query_start = i * queries_per_thread;
            thread_params[i].query_end = (i == num_threads - 1) ? n_query_data : (i + 1) * queries_per_thread;
            thread_params[i].n_queries_processed = thread_params[i].query_end - thread_params[i].query_start;
            
            // Pass RDMA resources.
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
        
        // Wait for threads to finish.
        for (int i = 0; i < num_threads; ++i) {
            void* status;
            int rc = pthread_join(threads[i], &status);
            if (rc) {
                std::cerr << "Error: unable to join thread, " << rc << std::endl;
                exit(-1);
            }
        }
        
        // Calculate average latency and throughput.
        double total_latency = 0.0;
        size_t total_queries_processed = 0;
        for (int i = 0; i < num_threads; ++i) {
            total_latency += thread_params[i].latency;
            total_queries_processed += thread_params[i].n_queries_processed;
        }
        double avg_latency = total_latency / total_queries_processed;
        double throughput = static_cast<double>(total_queries_processed) / (avg_latency / 1000000.0); // queries per second
        
        avg_latencies.push_back(avg_latency);
        throughputs.push_back(throughput);
        
        std::cout << "Threads: " << num_threads 
                  << ", Avg Latency (us): " << avg_latency 
                  << ", Throughput (q/s): " << throughput << std::endl;
        
        // Clean up remote QPs for the threads used in this test.
        for (int i = 0; i < num_threads; ++i) {
        cm.delete_remote_rc(unique_id + "-" + std::to_string(num_threads) + "-" + std::to_string(i), keys[i]);
    }
    } 
    
    // Write results to output file.
    std::ofstream outfile("../benchs/dhnsw/sift1M@10benchmark_throughput_latency.txt");
    outfile << "threads\tlatency(us)\tthroughput(q/s)" << std::endl;
    for (size_t i = 0; i < thread_counts.size(); ++i) {
        outfile << thread_counts[i] << "\t" << avg_latencies[i] << "\t" << throughputs[i] << std::endl;
    }
    outfile.close();
    
    return 0;
}

void* client_worker(void* param) {
    thread_param_t& thread_param = *(thread_param_t*)param;
    int thread_id = thread_param.thread_id;
    bind_thread_to_cores(thread_id, thread_param.core_start, thread_param.omp_threads_per_worker);
   
    omp_set_num_threads(thread_param.omp_threads_per_worker);

    

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
        
        // Prepare data pointers
        const float* query_data_ptr = query_data.data() + query_start * dim_query_data;
        const int* ground_truth_ptr = ground_truth.data() + query_start * dim_ground_truth;

        std::cout << "query_data_ptr: " << query_data_ptr << std::endl;
        std::cout << "ground_truth_ptr: " << ground_truth_ptr << std::endl;

        int branching_k = 5;
        int top_k = 1;
        // Fixed ef_search value of 128
        int ef = 48;

        std::vector<int> sub_hnsw_tosearch;
        // std::vector<std::vector<dhnsw_idx_t>> mapping = local_hnsw.get_local_mapping(); // For easy testing
        
        local_hnsw.set_meta_ef_search(ef);
        std::vector<int> local_sub_hnsw_tag;
        local_hnsw.set_local_sub_hnsw_tag(local_sub_hnsw_tag);

        // Perform sub-searches, fetching sub_hnsw over RDMA as needed
        float* meta_distances = new float[branching_k * n_query_data_thread];
        dhnsw_idx_t* meta_labels = new dhnsw_idx_t[branching_k * n_query_data_thread];
        dhnsw_idx_t* sub_hnsw_tags = new dhnsw_idx_t[top_k * n_query_data_thread];
        dhnsw_idx_t* labels = new dhnsw_idx_t[top_k * n_query_data_thread];
        float* distances = new float[top_k * n_query_data_thread];
        
        auto start = high_resolution_clock::now();
        local_hnsw.meta_search(n_query_data_thread, query_data_ptr, branching_k, meta_distances, meta_labels, sub_hnsw_tosearch);
          
        // naive search
        // std::pair<double, double> result = local_hnsw.sub_search_each_naive(n_query_data_thread, query_data_ptr, branching_k, top_k, distances, labels, sub_hnsw_tosearch, sub_hnsw_tags, ef, RDMA);
        
        // dhnsw search without db
        // std::pair<double, double> result = local_hnsw.sub_search_each(n_query_data_thread, query_data_ptr, branching_k, top_k, distances, labels, sub_hnsw_tosearch, sub_hnsw_tags, ef, RDMA);
        // dhnsw search with db
        std::pair<double, double> result = local_hnsw.sub_search_each(n_query_data_thread, query_data_ptr, branching_k, top_k, distances, labels, sub_hnsw_tosearch, sub_hnsw_tags, ef, RDMA_DOORBELL);
      
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start).count();

        thread_param.latency = static_cast<double>(duration);
        thread_param.throughput = n_query_data_thread;

        // Clean up
        delete[] labels;
        delete[] distances;
        delete[] sub_hnsw_tags;
        delete[] meta_distances;
        delete[] meta_labels;

        RDMA_LOG(4) << "Thread " << thread_id << " processed " << n_query_data_thread << " queries in " << duration << " us";

    } catch (const std::exception& e) {
        std::cerr << "Thread " << thread_id << " encountered an exception: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "Thread " << thread_id << " encountered an unknown exception." << std::endl;
        return nullptr;
    }
    return nullptr;
}