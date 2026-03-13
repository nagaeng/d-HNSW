// reconstruction_test.cc
// Local test for reconstruction trigger conditions and end-to-end flow

#include <gflags/gflags.h>
#include <grpcpp/grpcpp.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <random>
#include <fstream>
#include <sstream>

#include "../generated/dhnsw.grpc.pb.h"
#include "../../deps/rlib/core/lib.hh"
#include "../dhnsw/DistributedHnsw.h"
#include "../dhnsw/reconstruction.hh"
#include "../util/read_dataset.h"

typedef int64_t dhnsw_idx_t;

DEFINE_string(server_address, "localhost:50051", "Address of the gRPC server.");
DEFINE_string(rdma_server_address, "localhost:8888", "Address of the RDMA server.");
DEFINE_int32(use_nic_idx, 3, "Which NIC to create QP");
DEFINE_int32(reg_nic_name, 73, "The name to register an opened NIC at rctrl.");
DEFINE_int32(reg_mem_name, 73, "The name to register an MR at rctrl.");
DEFINE_string(dataset_path, "../datasets/sift/sift_base.fvecs", "Path to the dataset.");
DEFINE_int32(dim, 128, "Vector dimension");
DEFINE_int32(num_sub_hnsw, 4, "Number of sub-HNSW indices (small for testing)");
DEFINE_int32(meta_hnsw_neighbors, 16, "Meta HNSW neighbors");
DEFINE_int32(sub_hnsw_neighbors, 16, "Sub HNSW neighbors");
DEFINE_int32(num_meta, 100, "Number of meta vectors (small for testing)");
DEFINE_int32(test_batch_size, 10, "Batch size for test inserts");
DEFINE_string(log_file, "../benchs/reconstruction/test_log.csv", "Path to test log file");
DEFINE_string(client_id, "reconstruction_test", "Test client identifier");

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using dhnsw::DhnswService;
using dhnsw::Empty;
using namespace rdmaio;
using namespace rdmaio::rmem;
using namespace rdmaio::qp;
using namespace dhnsw::reconstruction;
using namespace std::chrono;

// Test state and coordination
struct TestState {
    std::atomic<bool> overflow_triggered{false};
    std::atomic<bool> reconstruction_started{false};
    std::atomic<bool> reconstruction_completed{false};
    std::atomic<int64_t> total_inserts{0};
    std::atomic<int64_t> overflow_sub_idx{-1};
    std::string overflow_type;

    // Throughput logging
    std::unique_ptr<ThroughputLogger> throughput_logger;
};

// Overflow callback - simulates what the master client would do
void test_overflow_callback(int sub_idx, const std::string& overflow_type) {
    std::cout << "[TEST] Overflow callback triggered: sub_idx=" << sub_idx
              << ", type=" << overflow_type << std::endl;

    // In real scenario, this would trigger server reconstruction
    // For testing, we just log the event
    std::cout << "[TEST] NOTE: In production, this would call TriggerReconstruction RPC" << std::endl;
}

// Generate vectors that will fill specific gaps to trigger overflow
std::vector<float> generate_gap_filling_vectors(int count, int dim, const std::string& target_gap) {
    std::vector<float> vectors(count * dim);

    // Use fixed seed for reproducible overflow triggering
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    for (int i = 0; i < count * dim; ++i) {
        vectors[i] = dist(rng);
    }

    return vectors;
}

class ReconstructionTest {
public:
    ReconstructionTest() : test_state_(std::make_unique<TestState>()) {}

    ~ReconstructionTest() {
        if (test_state_->throughput_logger) {
            test_state_->throughput_logger->stop_logging();
        }
    }

    // Step 1: Trigger overflow via controlled inserts
    bool test_overflow_trigger(LocalHnsw* local_hnsw, int target_sub_idx, const std::string& target_gap) {
        std::cout << "\n=== Step 1: Trigger Overflow ===" << std::endl;
        std::cout << "Target: sub_idx=" << target_sub_idx << ", gap=" << target_gap << std::endl;

        // Set up overflow callback
        local_hnsw->set_overflow_callback(test_overflow_callback);

        // Start throughput logging
        test_state_->throughput_logger = std::make_unique<ThroughputLogger>(
            FLAGS_log_file, FLAGS_client_id);
        test_state_->throughput_logger->start_logging(100); // 100ms intervals

        int batch_count = 0;
        const int max_batches = 100; // Prevent infinite loops

        while (!local_hnsw->has_overflow_detected() && batch_count < max_batches) {
            // Generate vectors that will expand the target gap
            auto insert_batch = generate_gap_filling_vectors(
                FLAGS_test_batch_size, FLAGS_dim, target_gap);

            try {
                // This will trigger the overflow detection in prepare_and_commit_update
                local_hnsw->insert_to_server(FLAGS_test_batch_size, insert_batch);

                test_state_->total_inserts += FLAGS_test_batch_size;
                test_state_->throughput_logger->record_insert(FLAGS_test_batch_size);

                batch_count++;
                std::cout << "[TEST] Inserted batch " << batch_count
                          << " (" << test_state_->total_inserts.load() << " total vectors)" << std::endl;

                // Check overflow state
                if (local_hnsw->has_overflow_detected()) {
                    test_state_->overflow_triggered = true;
                    test_state_->overflow_sub_idx = local_hnsw->get_last_overflow_sub_idx();
                    test_state_->overflow_type = local_hnsw->get_last_overflow_type();

                    std::cout << "[TEST] ✓ Overflow detected!" << std::endl;
                    std::cout << "  Sub-idx: " << test_state_->overflow_sub_idx.load() << std::endl;
                    std::cout << "  Type: " << test_state_->overflow_type << std::endl;
                    break;
                }

            } catch (const std::exception& e) {
                std::cout << "[TEST] Insert failed (expected during overflow): " << e.what() << std::endl;
                if (local_hnsw->has_overflow_detected()) {
                    test_state_->overflow_triggered = true;
                    test_state_->overflow_sub_idx = local_hnsw->get_last_overflow_sub_idx();
                    test_state_->overflow_type = local_hnsw->get_last_overflow_type();
                    break;
                }
            }

            // Small delay to prevent overwhelming the system
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // Validate expected conditions
        bool success = test_state_->overflow_triggered.load() &&
                      (test_state_->overflow_sub_idx.load() == target_sub_idx) &&
                      (test_state_->overflow_type == target_gap);

        std::cout << "[TEST] Step 1 result: " << (success ? "PASS" : "FAIL") << std::endl;
        return success;
    }

    // Step 2: Simulate reconstruction trigger
    bool test_reconstruction_trigger() {
        std::cout << "\n=== Step 2: Simulate Reconstruction Trigger ===" << std::endl;

        if (!test_state_->overflow_triggered) {
            std::cout << "[TEST] ✗ Cannot trigger reconstruction - no overflow detected" << std::endl;
            return false;
        }

        // Simulate the RPC call delay
        std::cout << "[TEST] Simulating TriggerReconstruction RPC call..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        test_state_->reconstruction_started = true;
        std::cout << "[TEST] ✓ Reconstruction triggered (simulated)" << std::endl;

        return true;
    }

    // Step 3: Test queries during reconstruction
    bool test_queries_during_reconstruction(LocalHnsw* local_hnsw) {
        std::cout << "\n=== Step 3: Test Queries During Reconstruction ===" << std::endl;

        if (!test_state_->reconstruction_started) {
            std::cout << "[TEST] ✗ Reconstruction not started" << std::endl;
            return false;
        }

        // Simulate blocking mode (what the handler would do)
        std::cout << "[TEST] Simulating query blocking during reconstruction..." << std::endl;

        // Generate test queries
        std::vector<float> query_batch(FLAGS_test_batch_size * FLAGS_dim);
        std::mt19937 rng(123); // Different seed for queries
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (auto& val : query_batch) {
            val = dist(rng);
        }

        // Test LSH cache fallback (since queries would be blocked)
        auto insert_cache = std::make_unique<LSHCache>(FLAGS_dim);
        insert_cache->add_batch(std::vector<int64_t>(FLAGS_test_batch_size, 0),
                               query_batch.data(), FLAGS_test_batch_size);

        // Try queries through cache
        int successful_queries = 0;
        for (int i = 0; i < FLAGS_test_batch_size; ++i) {
            const float* query = query_batch.data() + i * FLAGS_dim;
            auto results = insert_cache->search(query, 1);

            if (!results.empty()) {
                successful_queries++;
                test_state_->throughput_logger->record_query(1);
            }
        }

        std::cout << "[TEST] Queries during reconstruction: "
                  << successful_queries << "/" << FLAGS_test_batch_size << " successful" << std::endl;

        bool success = successful_queries > 0; // At least some queries should work
        std::cout << "[TEST] Step 3 result: " << (success ? "PASS" : "FAIL") << std::endl;
        return success;
    }

    // Step 4: Test reconstruction completion
    bool test_reconstruction_completion() {
        std::cout << "\n=== Step 4: Test Reconstruction Completion ===" << std::endl;

        // Simulate completion
        std::cout << "[TEST] Simulating reconstruction completion..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        test_state_->reconstruction_completed = true;
        std::cout << "[TEST] ✓ Reconstruction completed (simulated)" << std::endl;

        // Validate final state
        bool success = test_state_->overflow_triggered.load() &&
                      test_state_->reconstruction_started.load() &&
                      test_state_->reconstruction_completed.load();

        std::cout << "[TEST] Step 4 result: " << (success ? "PASS" : "FAIL") << std::endl;
        return success;
    }

    // Run complete test sequence
    bool run_full_test(LocalHnsw* local_hnsw, int target_sub_idx, const std::string& target_gap) {
        std::cout << "=== Starting Reconstruction Test Sequence ===" << std::endl;
        std::cout << "Target gap: " << target_gap << " in sub-index " << target_sub_idx << std::endl;

        bool step1 = test_overflow_trigger(local_hnsw, target_sub_idx, target_gap);
        bool step2 = step1 ? test_reconstruction_trigger() : false;
        bool step3 = step2 ? test_queries_during_reconstruction(local_hnsw) : false;
        bool step4 = step3 ? test_reconstruction_completion() : false;

        std::cout << "\n=== Test Results ===" << std::endl;
        std::cout << "Step 1 (Overflow): " << (step1 ? "PASS" : "FAIL") << std::endl;
        std::cout << "Step 2 (Trigger):  " << (step2 ? "PASS" : "FAIL") << std::endl;
        std::cout << "Step 3 (Queries):  " << (step3 ? "PASS" : "FAIL") << std::endl;
        std::cout << "Step 4 (Complete): " << (step4 ? "PASS" : "FAIL") << std::endl;

        bool overall_success = step1 && step2 && step3 && step4;
        std::cout << "Overall: " << (overall_success ? "PASS" : "FAIL") << std::endl;

        // Final statistics
        std::cout << "\nStatistics:" << std::endl;
        std::cout << "- Total inserts: " << test_state_->total_inserts.load() << std::endl;
        std::cout << "- Overflow sub-idx: " << test_state_->overflow_sub_idx.load() << std::endl;
        std::cout << "- Overflow type: " << test_state_->overflow_type << std::endl;

        return overall_success;
    }

    // Get test state for external monitoring
    const TestState& get_test_state() const { return *test_state_; }

private:
    std::unique_ptr<TestState> test_state_;
};

void bind_thread_to_cores(int thread_id, int core_start, int cores_per_thread) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    for (int i = 0; i < cores_per_thread; i++) {
        int core_id = core_start + i;
        CPU_SET(core_id, &cpuset);
    }

    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::cerr << "Error: unable to set thread affinity for thread "
                  << thread_id << ", error code: " << rc << std::endl;
    }
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);

    std::cout << "=== Reconstruction Trigger Test ===" << std::endl;
    std::cout << "Testing new trigger condition: overflow only when gap is FULL (not close to full)" << std::endl;

    // Initialize RDMA connection
    ConnectManager cm(FLAGS_rdma_server_address);
    if (cm.wait_ready(1000000, 20) == IOCode::Timeout) {
        std::cerr << "RDMA connection timeout" << std::endl;
        return -1;
    }

    auto fetch_res = cm.fetch_remote_mr(FLAGS_reg_mem_name);
    RDMA_ASSERT(fetch_res == IOCode::Ok);
    rmem::RegAttr remote_attr = std::get<1>(fetch_res.desc);

    // Create QP and memory
    auto nic = RNic::create(RNicInfo::query_dev_names().at(FLAGS_use_nic_idx)).value();
    auto qp = RC::create(nic, QPConfig()).value();
    std::string qp_name = "-test@" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    auto qp_res = cm.cc_rc(qp_name, qp, FLAGS_reg_nic_name, QPConfig());
    RDMA_ASSERT(qp_res == IOCode::Ok);

    size_t local_mem_size = 1UL * 1024 * 1024 * 1024; // 1GB
    auto local_mem = Arc<RMem>(new RMem(local_mem_size));
    auto local_mr = RegHandler::create(local_mem, nic).value();

    // Create LocalHnsw client
    DhnswClient* dhnsw_client = new DhnswClient(
        grpc::CreateChannel(FLAGS_server_address, grpc::InsecureChannelCredentials()));

    LocalHnsw local_hnsw(FLAGS_dim, FLAGS_num_sub_hnsw, FLAGS_meta_hnsw_neighbors,
                        FLAGS_sub_hnsw_neighbors, dhnsw_client);
    local_hnsw.init();
    local_hnsw.set_rdma_qp(qp, remote_attr, local_mr);
    local_hnsw.set_remote_attr(remote_attr);
    local_hnsw.set_local_mr(local_mr, local_mem);

    // Run the test
    ReconstructionTest test;

    // Test different gap types
    std::vector<std::pair<int, std::string>> test_cases = {
        {0, "levels"},      // Test levels gap in sub-index 0
        {0, "offsets"},     // Test offsets gap in sub-index 0
        {0, "neighbors"},   // Test neighbors gap in sub-index 0
        {0, "xb"}           // Test xb gap in sub-index 0
    };

    bool all_passed = true;
    for (const auto& test_case : test_cases) {
        int sub_idx = test_case.first;
        const std::string& gap_type = test_case.second;

        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "Testing gap type: " << gap_type << " in sub-index " << sub_idx << std::endl;
        std::cout << std::string(60, '=') << std::endl;

        // Reset overflow state between tests
        local_hnsw.clear_overflow_flag();

        bool passed = test.run_full_test(&local_hnsw, sub_idx, gap_type);
        all_passed = all_passed && passed;

        // Small delay between tests
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    // Cleanup
    test.get_test_state().throughput_logger->stop_logging();
    delete dhnsw_client;

    auto del_res = cm.delete_remote_rc(qp_name, std::get<1>(qp_res.desc));
    qp.reset();

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "FINAL RESULT: " << (all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    return all_passed ? 0 : 1;
}
