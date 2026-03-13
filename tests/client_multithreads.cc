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

typedef int64_t dhnsw_idx_t;

DEFINE_string(server_address, "130.127.134.42:50051", "Address of the gRPC server.");
DEFINE_string(rdma_server_address, "192.168.1.2:8888", "Address of the RDMA server.");
DEFINE_int32(use_nic_idx, 3, "Which NIC to create QP");
// 0 for r320 3 for r650
DEFINE_int32(reg_nic_name, 73, "The name to register an opened NIC at rctrl.");
DEFINE_int32(reg_mem_name, 73, "The name to register an MR at rctrl.");
DEFINE_string(query_data_path, "../datasets/sift/sift_query.fvecs", "Path to the query data.");
DEFINE_string(ground_truth_path, "../datasets/sift/sift_groundtruth.ivecs", "Path to the ground truth data.");

// DEFINE_string(query_data_path, "../datasets/gist/gist_query.fvecs", "Path to the query data.");
// DEFINE_string(ground_truth_path, "../datasets/gist/gist_groundtruth.ivecs", "Path to the ground truth data.");

DEFINE_string(client_id, "", "Unique client identifier for multi-node experiments (e.g., node21, node22). Auto-generated from PID+timestamp if empty.");

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
};
void* client_worker(void* param);

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

    // Sample 1/3 of the query data
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
    n_query_data = query_data.size() / dim_query_data;

    int num_threads = std::thread::hardware_concurrency() > 10 ? 10 : std::thread::hardware_concurrency();
    int omp_threads_per_worker = std::thread::hardware_concurrency() / num_threads;
    std::vector<pthread_t> threads(num_threads);
    std::vector<thread_param_t> thread_params(num_threads);

    // --- Generate unique client ID for multi-node QP naming ---
    std::string unique_id = FLAGS_client_id;
    if (unique_id.empty()) {
        auto ts = std::chrono::system_clock::now().time_since_epoch().count();
        unique_id = std::to_string(getpid()) + "-" + std::to_string(ts);
    }
    std::cout << "Client unique ID: " << unique_id << std::endl;

    // Initialize RDMA resources in the main function
    // Create a ConnectManager and connect to the server's controller
    ConnectManager cm(FLAGS_rdma_server_address);
    if (cm.wait_ready(1000000, 20) == IOCode::Timeout) {
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
        std::string qp_name = unique_id + "-qp-" + std::to_string(i);
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

    // Split queries among threads
    int queries_per_thread = n_query_data / num_threads;
    
    for (int i = 0; i < num_threads; ++i) {
        thread_params[i].thread_id = i;
        thread_params[i].throughput = 0;
        thread_params[i].latency = 0;
        if(i == num_threads){
            thread_params[i].omp_threads_per_worker = omp_threads_per_worker + std::thread::hardware_concurrency() % num_threads;
        } else{
        thread_params[i].omp_threads_per_worker = omp_threads_per_worker;
        }
        // Assign per-thread query ranges
        thread_params[i].query_start = i * queries_per_thread;
        if (i == num_threads -1) {
            // Last thread takes the remaining queries
            thread_params[i].query_end = n_query_data;
        } else {
            thread_params[i].query_end = (i+1) * queries_per_thread;
        }

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
            std::cerr << "Error: unable to join," << rc << std::endl;
            exit(-1);
        }
    }


    // Aggregate results per ef value
    std::vector<int> ef_search_values = {1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128};
    std::vector<int> ef_values;
    std::vector<float> avg_recalls;
    std::vector<double> avg_latencies;

    for (size_t ef_idx = 0; ef_idx < ef_search_values.size(); ++ef_idx) {
        int ef = ef_search_values[ef_idx];
        ef_values.push_back(ef);
        float sum_recalls = 0.0f;
        double sum_latency = 0.0;
        std::cout << "EF: " << ef << std::endl;
        for (int t = 0; t < num_threads; ++t) {
            std::cout << "Thread " << t << " recall: " << thread_params[t].per_ef_recalls[ef_idx] << std::endl;
            std::cout << "Thread " << t << " latency: " << thread_params[t].per_ef_latencies[ef_idx] << std::endl;
            sum_recalls += thread_params[t].per_ef_recalls[ef_idx];
            sum_latency += thread_params[t].per_ef_latencies[ef_idx];
        }
        avg_recalls.push_back(sum_recalls / num_threads);
        avg_latencies.push_back(sum_latency / num_threads);
    }

    // Write results to output file
    std::ofstream outfile("../benchs/dhnsw_withoutdb/sift1M@10benchmark_results_test.txt");
    outfile << "latency(us)\trecall" << std::endl;
    for (size_t i = 0; i < ef_values.size(); ++i) {
        outfile << avg_latencies[i] << ",\t" << avg_recalls[i] << std::endl;
    }
    outfile.close();
    // Clean up RDMA resources 
    for (int i = 0; i < num_threads; ++i) {
        
        std::string qp_name = unique_id + "-qp-" + std::to_string(i);
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
    omp_set_num_threads(thread_param.omp_threads_per_worker);
    // Initialize per-ef recall and latency vectors immediately
    std::vector<int> ef_search_values = {1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128};
    thread_param.per_ef_recalls.resize(ef_search_values.size(), 0.0f);
    thread_param.per_ef_latencies.resize(ef_search_values.size(), 0.0);

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

        int branching_k = 10;
        int top_k = 10;

        std::vector<int> sub_hnsw_tosearch;
        std::vector<std::vector<dhnsw_idx_t>> mapping = local_hnsw.get_local_mapping(); // For easy testing
        for (size_t ef_idx = 0; ef_idx < ef_search_values.size(); ++ef_idx) {
            int ef = ef_search_values[ef_idx];
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
            local_hnsw.sub_search_each(n_query_data_thread, query_data_ptr, branching_k, top_k, distances, labels, sub_hnsw_tosearch, sub_hnsw_tags, ef, RDMA);
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<microseconds>(stop - start).count();

            thread_param.per_ef_latencies[ef_idx] = static_cast<double>(duration);

            dhnsw_idx_t *original_index = new dhnsw_idx_t[top_k * n_query_data_thread];
            for(int i = 0; i < n_query_data_thread; i++) {
                for(int j = 0; j < top_k; j++) {
                    original_index[i * top_k + j] = mapping[sub_hnsw_tags[i * top_k + j]][labels[i * top_k + j]];
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
                    if (ground_truth_set.find(retrieved_idx) != ground_truth_set.end()) {
                        correct++;
                    }
                }
            

            recall = static_cast<float>(correct) / (n_query_data_thread * top_k); 

            thread_param.per_ef_recalls[ef_idx] = recall;
    
            }
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
        // thread_param.latency = std::accumulate(thread_param.per_ef_latencies.begin(), thread_param.per_ef_latencies.end(), 0.0);

        RDMA_LOG(4) << "Thread " << thread_id << " returns";

    } catch (const std::exception& e) {
        std::cerr << "Thread " << thread_id << " encountered an exception: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        std::cerr << "Thread " << thread_id << " encountered an unknown exception." << std::endl;
        return nullptr;
    }

    return nullptr;
}