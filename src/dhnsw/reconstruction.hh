#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <chrono>
#include <random>
#include <functional>
#include <queue>
#include <fstream>
#include <memory>
#include <grpcpp/grpcpp.h>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>

// Forward declarations
namespace faiss {
    class IndexHNSWFlat;
}

namespace dhnsw {
namespace reconstruction {

using Clock = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;

//==============================================================================
// LSH-based Insert Cache for fast approximate lookup during reconstruction
//==============================================================================
class LSHCache {
public:
    LSHCache(int dimension, int num_hash_tables = 8, int num_hash_bits = 12);
    ~LSHCache() = default;

    // Add a vector to the cache
    void add(int64_t id, const float* vector);
    
    // Add multiple vectors
    void add_batch(const std::vector<int64_t>& ids, const float* vectors, int count);
    
    // Find approximate nearest neighbors in the cache
    std::vector<std::pair<int64_t, float>> search(const float* query, int k) const;
    
    // Get all cached vectors
    std::vector<float> get_all_vectors() const;
    std::vector<int64_t> get_all_ids() const;
    
    // Get cache info
    size_t size() const { return vectors_.size() / dim_; }
    int dimension() const { return dim_; }
    
    // Clear the cache
    void clear();

    // Reinitialize with new dimensions (for deserialization)
    void reinitialize(int new_dim, int new_num_tables, int new_num_bits);

    // Serialize/deserialize for network transfer
    std::vector<uint8_t> serialize() const;
    static void deserialize_into(const std::vector<uint8_t>& data, LSHCache& cache);

private:
    int dim_;
    int num_tables_;
    int num_bits_;
    
    // Random projection matrices for LSH
    std::vector<std::vector<float>> projection_matrices_;
    
    // Hash tables: bucket_id -> list of vector indices
    std::vector<std::unordered_map<uint64_t, std::vector<size_t>>> hash_tables_;
    
    // Stored vectors and their IDs
    std::vector<float> vectors_;
    std::vector<int64_t> ids_;
    
    mutable std::mutex mutex_;
    
    // Compute LSH hash for a vector
    uint64_t compute_hash(const float* vector, int table_idx) const;
    
    // Initialize random projections
    void init_projections();
};


//==============================================================================
// Throughput Logger for monitoring reconstruction impact
//==============================================================================
struct ThroughputSample {
    uint64_t timestamp_ms;
    double qps;           // Queries per second
    double ips;           // Inserts per second
    double total_ops;     // Total operations per second
    bool is_blocked;      // Whether queries were blocked
    std::string client_id;
    std::string event;    // Special events (optional)
};

// Batch-level record for accurate throughput over time tracking
struct BatchRecord {
    int worker_id;              // Worker thread ID
    uint64_t batch_start_ms;    // Batch start timestamp (milliseconds)
    uint64_t batch_end_ms;      // Batch end timestamp (milliseconds)
    int query_count;            // Number of queries in this batch
    int insert_count;           // Number of inserts in this batch
    bool blocked;               // Whether batch was blocked during reconstruction
    std::string client_id;
};

class ThroughputLogger {
public:
    ThroughputLogger(const std::string& output_file, const std::string& client_id);
    ~ThroughputLogger();

    // Record operation counts (legacy support for periodic sampling)
    void record_query(int count = 1);
    void record_insert(int count = 1);
    
    // Record batch-level operations for accurate throughput tracking
    void record_batch(int worker_id, uint64_t batch_start_ms, uint64_t batch_end_ms,
                     int query_count, int insert_count, bool blocked = false);
    
    // Set blocked state
    void set_blocked(bool blocked);

    // Record special events
    void record_event(const std::string& event);

    // Get current throughput
    ThroughputSample get_current_sample() const;
    
    // Flush samples to file
    void flush();
    
    // Start/stop background logging
    void start_logging(int interval_ms = 100);
    void stop_logging();

private:
    std::string output_file_;
    std::string client_id_;
    
    std::atomic<int64_t> query_count_{0};
    std::atomic<int64_t> insert_count_{0};
    std::atomic<bool> is_blocked_{false};
    
    TimePoint last_sample_time_;
    int64_t last_query_count_{0};
    int64_t last_insert_count_{0};
    
    std::vector<ThroughputSample> samples_;
    std::vector<BatchRecord> batch_records_;
    mutable std::mutex mutex_;
    
    std::thread logging_thread_;
    std::atomic<bool> stop_logging_{false};
    
    void logging_worker(int interval_ms);
};


//==============================================================================
// Reconstruction State Machine for coordination
//==============================================================================
enum class ReconstructionPhase {
    IDLE,                    // No reconstruction in progress
    TRIGGERED,               // Reconstruction triggered, waiting for server
    BUILDING_INDEX,          // Server is rebuilding index
    COPYING_DATA,            // Copying rebuilt index to RDMA memory
    NOTIFYING_CLIENTS,       // Notifying clients of completion
    WAITING_FOR_ACKS,        // Waiting for all clients to acknowledge
    COMPLETED                // Reconstruction complete
};

struct ReconstructionState {
    ReconstructionPhase phase{ReconstructionPhase::IDLE};
    uint64_t reconstruction_id{0};
    double progress{0.0};
    std::string message;
    TimePoint start_time;
    TimePoint end_time;
    
    // New index data
    std::vector<uint8_t> new_meta_hnsw;
    std::vector<uint64_t> new_offset_subhnsw;
    std::vector<uint64_t> new_offset_para;
    std::vector<uint64_t> new_overflow;
    std::vector<std::vector<int64_t>> new_mapping;
    uint64_t new_rdma_offset{0};
};


//==============================================================================
// Client-side Reconstruction Handler
//==============================================================================
class ClientReconstructionHandler {
public:
    using UpdateCallback = std::function<void(const ReconstructionState&)>;
    
    ClientReconstructionHandler(const std::string& client_id, bool is_master);
    ~ClientReconstructionHandler();

    // Set callback for when reconstruction completes
    void set_update_callback(UpdateCallback callback) { 
        update_callback_ = callback; 
    }
    
    // Check if queries should be blocked
    bool should_block_queries() const { return block_queries_.load(); }
    
    // Enter/exit blocking mode
    void enter_blocking_mode();
    void exit_blocking_mode();
    
    // Wait for reconstruction to complete (with timeout)
    bool wait_for_completion(int timeout_ms = 30000);
    
    // Handle reconstruction notification from server
    void handle_notification(const ReconstructionState& state);
    
    // Get the insert cache (for master node)
    LSHCache* get_insert_cache() { return &insert_cache_; }
    
    // Synchronize insert cache from master (for worker nodes)
    void sync_insert_cache(const std::vector<uint8_t>& cache_data);

    // Get current state
    ReconstructionPhase get_phase() const { return current_state_.phase; }

private:
    std::string client_id_;
    bool is_master_;
    
    ReconstructionState current_state_;
    std::atomic<bool> block_queries_{false};
    
    LSHCache insert_cache_;
    
    UpdateCallback update_callback_;
    
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};


//==============================================================================
// Server-side Reconstruction Manager
//==============================================================================
class ServerReconstructionManager {
public:
    ServerReconstructionManager(
        int dimension,
        int num_sub_hnsw,
        int meta_hnsw_neighbors,
        int sub_hnsw_neighbors,
        int num_meta
    );
    ~ServerReconstructionManager();

    // Check if reconstruction is needed based on overflow status
    bool needs_reconstruction(const std::vector<uint64_t>& overflow) const;
    
    // Trigger reconstruction with buffered inserts
    uint64_t start_reconstruction(
        const std::vector<float>& existing_data,
        const std::vector<float>& buffered_inserts,
        int existing_count,
        int insert_count
    );
    
    // Get current reconstruction status
    ReconstructionState get_status() const;
    
    // Get the rebuilt index data (after reconstruction completes)
    bool get_rebuilt_data(
        std::vector<uint8_t>& meta_hnsw,
        std::vector<uint64_t>& offset_subhnsw,
        std::vector<uint64_t>& offset_para,
        std::vector<uint64_t>& overflow,
        std::vector<std::vector<int64_t>>& mapping,
        std::vector<uint8_t>& serialized_data
    );
    
    // Client registration for notifications
    void register_client(const std::string& client_id, const std::string& address, bool is_master);
    void unregister_client(const std::string& client_id);
    
    // Client acknowledgment
    bool acknowledge_completion(const std::string& client_id, uint64_t reconstruction_id);
    bool all_clients_acknowledged() const;
    
    // Set threshold for overflow (percentage of gap used before triggering)
    void set_overflow_threshold(double threshold) { overflow_threshold_ = threshold; }

    // Force state transitions (called by coordinator after ACK wait completes)
    void force_complete();
    void force_idle();

private:
    int dim_;
    int num_sub_hnsw_;
    int meta_M_;
    int sub_M_;
    int num_meta_;
    
    double overflow_threshold_{0.8};  // Trigger at 80% overflow by default
    
    ReconstructionState state_;
    mutable std::mutex state_mutex_;
    
    // Registered clients
    struct ClientInfo {
        std::string address;
        bool is_master;
        bool acknowledged;
    };
    std::unordered_map<std::string, ClientInfo> clients_;
    mutable std::mutex clients_mutex_;
    
    // Background reconstruction thread
    std::thread reconstruction_thread_;
    std::atomic<bool> stop_reconstruction_{false};
    
    // Internal data after reconstruction
    std::vector<uint8_t> rebuilt_serialized_data_;
    
    void reconstruction_worker(
        std::vector<float> existing_data,
        std::vector<float> buffered_inserts,
        int existing_count,
        int insert_count
    );
};


//==============================================================================
// Query Blocker - RAII-style query blocking during reconstruction
//==============================================================================
class QueryBlocker {
public:
    QueryBlocker(ClientReconstructionHandler& handler)
        : handler_(handler) {
        handler_.enter_blocking_mode();
    }
    
    ~QueryBlocker() {
        handler_.exit_blocking_mode();
    }
    
    // Non-copyable
    QueryBlocker(const QueryBlocker&) = delete;
    QueryBlocker& operator=(const QueryBlocker&) = delete;

private:
    ClientReconstructionHandler& handler_;
};


//==============================================================================
// Epoch-based RDMA Buffer Manager for safe reconstruction
//==============================================================================
class EpochBufferManager {
public:
    struct EpochBuffer {
        uint64_t epoch;
        uint64_t rdma_offset;      // Offset within the RDMA region
        size_t data_size;
        std::atomic<int> active_readers{0};  // Count of active readers
        bool is_valid{false};
    };

    EpochBufferManager(size_t total_rdma_size, uint8_t* rdma_base)
        : total_size_(total_rdma_size)
        , rdma_base_(rdma_base)
        , current_epoch_(0) {
        // Initialize with two buffer slots for double-buffering
        buffers_[0] = std::make_unique<EpochBuffer>();
        buffers_[1] = std::make_unique<EpochBuffer>();
    }

    // Initialize the initial epoch (epoch 0) with its buffer info
    void init_epoch_zero(uint64_t rdma_offset, size_t data_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        buffers_[0]->epoch = 0;
        buffers_[0]->rdma_offset = rdma_offset;
        buffers_[0]->data_size = data_size;
        buffers_[0]->is_valid = true;
    }

    // Acquire a read reference for the current epoch
    // Returns the epoch and base offset for RDMA reads
    std::pair<uint64_t, uint64_t> acquire_read() {
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t epoch = current_epoch_.load();
        int slot = epoch % 2;
        buffers_[slot]->active_readers.fetch_add(1, std::memory_order_acquire);
        return {epoch, buffers_[slot]->rdma_offset};
    }

    // Release a read reference
    void release_read(uint64_t epoch) {
        int slot = epoch % 2;
        int prev = buffers_[slot]->active_readers.fetch_sub(1, std::memory_order_release);
        if (prev == 1) {
            // Last reader - notify waiters
            std::lock_guard<std::mutex> lock(mutex_);
            cv_.notify_all();
        }
    }

    // Prepare a new buffer for writing (returns the write offset)
    // This blocks until the old buffer has no readers
    uint64_t prepare_new_epoch(size_t new_data_size) {
        std::unique_lock<std::mutex> lock(mutex_);

        uint64_t old_epoch = current_epoch_.load();
        uint64_t new_epoch = old_epoch + 1;
        int old_slot = old_epoch % 2;
        int new_slot = new_epoch % 2;

        // Wait for all readers on the NEW slot (which we're about to reuse) to finish
        // The new_slot was the previous-previous epoch's buffer
        cv_.wait(lock, [this, new_slot]() {
            return buffers_[new_slot]->active_readers.load(std::memory_order_acquire) == 0;
        });

        // Calculate new offset - use the opposite half of the buffer
        uint64_t new_offset;
        if (buffers_[old_slot]->rdma_offset == 0) {
            new_offset = total_size_ / 2;  // Use second half
        } else {
            new_offset = 0;  // Use first half
        }

        // Prepare the new buffer metadata
        buffers_[new_slot]->epoch = new_epoch;
        buffers_[new_slot]->rdma_offset = new_offset;
        buffers_[new_slot]->data_size = new_data_size;
        buffers_[new_slot]->is_valid = false;  // Not valid until commit

        return new_offset;
    }

    // Commit the new epoch after data has been written
    void commit_new_epoch() {
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t new_epoch = current_epoch_.load() + 1;
        int new_slot = new_epoch % 2;
        buffers_[new_slot]->is_valid = true;
        current_epoch_.fetch_add(1, std::memory_order_release);
    }

    // Wait for all readers on old epoch to finish
    bool wait_for_old_readers_quiesce(int timeout_seconds = 30) {
        std::unique_lock<std::mutex> lock(mutex_);
        uint64_t old_epoch = current_epoch_.load() - 1;
        if (old_epoch == static_cast<uint64_t>(-1)) return true;  // No old epoch

        int old_slot = old_epoch % 2;
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_seconds);

        return cv_.wait_until(lock, deadline, [this, old_slot]() {
            return buffers_[old_slot]->active_readers.load(std::memory_order_acquire) == 0;
        });
    }

    uint64_t get_current_epoch() const {
        return current_epoch_.load(std::memory_order_acquire);
    }

    uint64_t get_current_offset() const {
        std::lock_guard<std::mutex> lock(mutex_);
        int slot = current_epoch_.load() % 2;
        return buffers_[slot]->rdma_offset;
    }

    int get_active_readers(uint64_t epoch) const {
        int slot = epoch % 2;
        return buffers_[slot]->active_readers.load(std::memory_order_acquire);
    }

private:
    size_t total_size_;
    uint8_t* rdma_base_;
    std::atomic<uint64_t> current_epoch_;
    std::unique_ptr<EpochBuffer> buffers_[2];
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

// RAII guard for epoch-based reads
class EpochReadGuard {
public:
    EpochReadGuard(EpochBufferManager& manager)
        : manager_(manager) {
        auto [epoch, offset] = manager_.acquire_read();
        epoch_ = epoch;
        base_offset_ = offset;
    }

    ~EpochReadGuard() {
        manager_.release_read(epoch_);
    }

    uint64_t epoch() const { return epoch_; }
    uint64_t base_offset() const { return base_offset_; }

    // Non-copyable
    EpochReadGuard(const EpochReadGuard&) = delete;
    EpochReadGuard& operator=(const EpochReadGuard&) = delete;

private:
    EpochBufferManager& manager_;
    uint64_t epoch_;
    uint64_t base_offset_;
};

//==============================================================================
// Utility functions
//==============================================================================

// Get current timestamp in milliseconds
inline uint64_t current_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        Clock::now().time_since_epoch()
    ).count();
}

// Calculate overflow usage percentage
// Handles bidirectional shared-gap invariant:
// - Even indices (0, 2, 4, ...): grow FORWARD from gap_start toward gap_max
//   overflow[i*3] = gap_start (lower addr), overflow[i*3+2] = gap_max (higher addr)
// - Odd indices (1, 3, 5, ...): grow BACKWARD from gap_start toward gap_max
//   overflow[i*3] = gap_start (higher addr), overflow[i*3+2] = gap_max (lower addr)
inline double calculate_overflow_usage(
    const std::vector<uint64_t>& overflow,
    int sub_idx
) {
    if (overflow.size() < static_cast<size_t>((sub_idx + 1) * 3)) return 0.0;

    uint64_t slot0 = overflow[sub_idx * 3];      // gap_start (direction-dependent)
    uint64_t slot1 = overflow[sub_idx * 3 + 1];  // gap_current
    uint64_t slot2 = overflow[sub_idx * 3 + 2];  // gap_max (direction-dependent)

    // Determine growth direction based on even/odd index
    bool is_forward = (sub_idx % 2 == 0);

    if (is_forward) {
        // Even index: grows forward, slot0 < slot2
        // gap_start = slot0, gap_max = slot2
        if (slot2 <= slot0) return 0.0;  // No gap allocated yet
        uint64_t total = slot2 - slot0;
        uint64_t used = (slot1 >= slot0) ? (slot1 - slot0) : 0;
        return static_cast<double>(used) / static_cast<double>(total);
    } else {
        // Odd index: grows backward, slot0 > slot2
        // gap_start = slot0 (higher addr), gap_max = slot2 (lower addr)
        if (slot0 <= slot2) return 0.0;  // No gap allocated yet
        uint64_t total = slot0 - slot2;
        uint64_t used = (slot0 >= slot1) ? (slot0 - slot1) : 0;
        return static_cast<double>(used) / static_cast<double>(total);
    }
}

// Check if any sub-HNSW overflow is above threshold
inline bool any_overflow_above_threshold(
    const std::vector<uint64_t>& overflow,
    int num_sub_hnsw,
    double threshold = 0.8
) {
    for (int i = 0; i < num_sub_hnsw; ++i) {
        if (calculate_overflow_usage(overflow, i) >= threshold) {
            return true;
        }
    }
    return false;
}

//==============================================================================
// Global Append-Only Gap Buffer for Incremental Inserts
// This reduces reconstruction frequency by providing extra capacity
//==============================================================================
class GlobalGapBuffer {
public:
    // Per-sub-index spillover tracking
    // When a sub-index's local gap is full, vectors spill here
    struct SpilloverEntry {
        int sub_idx;
        int64_t vector_id;
        uint64_t data_offset;  // Offset within global gap
    };

    GlobalGapBuffer() 
        : gap_start_(0), gap_current_(0), gap_end_(0), dimension_(0)
        , vector_count_(0), is_full_(false) {}
    
    // Initialize the gap
    void init(uint64_t start, uint64_t end, int dim) {
        std::lock_guard<std::mutex> lock(spillover_mutex_);
        gap_start_ = start;
        gap_current_ = start;
        gap_end_ = end;
        dimension_ = dim;
        vector_count_.store(0);
        is_full_.store(false);
        spillover_entries_.clear();
    }
    
    // Get available capacity
    size_t available_bytes() const {
        uint64_t current = gap_current_;
        return (current < gap_end_) ? (gap_end_ - current) : 0;
    }
    
    size_t available_vectors() const {
        if (dimension_ == 0) return 0;
        return available_bytes() / (dimension_ * sizeof(float));
    }
    
    double usage_ratio() const {
        if (gap_end_ <= gap_start_) return 0.0;
        return static_cast<double>(gap_current_ - gap_start_) / 
               static_cast<double>(gap_end_ - gap_start_);
    }
    
    bool is_full() const { return is_full_.load(); }
    int vector_count() const { return vector_count_.load(); }
    int dimension() const { return dimension_; }
    uint64_t gap_start() const { return gap_start_; }
    uint64_t gap_current() const { return gap_current_; }
    uint64_t gap_end() const { return gap_end_; }
    
    // Try to allocate space for a vector
    // Returns the offset if successful, or 0 if full
    uint64_t try_allocate(size_t bytes) {
        std::lock_guard<std::mutex> lock(spillover_mutex_);
        if (gap_current_ + bytes > gap_end_) {
            is_full_.store(true);
            return 0;  // No space
        }
        uint64_t offset = gap_current_;
        gap_current_ += bytes;
        return offset;
    }
    
    // Add a spillover entry
    void add_spillover(int sub_idx, int64_t vector_id, uint64_t data_offset) {
        std::lock_guard<std::mutex> lock(spillover_mutex_);
        spillover_entries_.push_back({sub_idx, vector_id, data_offset});
        vector_count_.fetch_add(1);
    }
    
    // Get all spillover entries for reconstruction
    std::vector<SpilloverEntry> get_spillover_entries() const {
        std::lock_guard<std::mutex> lock(spillover_mutex_);
        return spillover_entries_;
    }
    
    // Clear the gap (after reconstruction)
    void clear() {
        std::lock_guard<std::mutex> lock(spillover_mutex_);
        gap_current_ = gap_start_;
        spillover_entries_.clear();
        vector_count_.store(0);
        is_full_.store(false);
    }

private:
    // Gap location in RDMA buffer
    uint64_t gap_start_;
    uint64_t gap_current_;  // Current write position
    uint64_t gap_end_;      // Maximum boundary
    int dimension_;
    
    // Gap metadata
    std::atomic<int> vector_count_;
    std::atomic<bool> is_full_;
    
    std::vector<SpilloverEntry> spillover_entries_;
    mutable std::mutex spillover_mutex_;
};

//==============================================================================
// Tiered Insert Buffer: Sub-gap -> Global gap -> Reconstruction
//==============================================================================
class TieredInsertBuffer {
public:
    TieredInsertBuffer(int num_sub_hnsw, int dimension, double gap_ratio = 0.1)
        : num_sub_hnsw_(num_sub_hnsw)
        , dimension_(dimension)
        , gap_ratio_(gap_ratio) {}
    
    // Set the global gap location after index serialization
    void set_global_gap(uint64_t start, uint64_t end) {
        global_gap_.init(start, end, dimension_);
    }
    
    // Check if reconstruction is needed
    // Only triggers when BOTH sub-gap is full AND global gap is full
    bool needs_reconstruction(
        const std::vector<uint64_t>& overflow,
        double threshold = 0.8
    ) const {
        // Check if any sub-gap is full
        bool any_sub_gap_full = false;
        for (int i = 0; i < num_sub_hnsw_; ++i) {
            if (calculate_overflow_usage(overflow, i) >= threshold) {
                any_sub_gap_full = true;
                break;
            }
        }
        
        if (!any_sub_gap_full) {
            return false;  // Still have sub-gap capacity
        }
        
        // Check if global gap is also full
        return global_gap_.usage_ratio() >= threshold;
    }
    
    // Try to insert into global gap (spillover from sub-gap)
    bool try_spillover_insert(
        int sub_idx,
        int64_t vector_id,
        const float* vector_data,
        uint8_t* rdma_buffer
    ) {
        size_t bytes_needed = dimension_ * sizeof(float);
        uint64_t offset = global_gap_.try_allocate(bytes_needed);
        
        if (offset == 0) {
            return false;  // Global gap is full
        }
        
        // Copy vector data to RDMA buffer at allocated offset
        std::memcpy(rdma_buffer + offset, vector_data, bytes_needed);
        
        // Track the spillover
        global_gap_.add_spillover(sub_idx, vector_id, offset);
        
        return true;
    }
    
    // Get global gap for reconstruction
    GlobalGapBuffer& get_global_gap() { return global_gap_; }
    const GlobalGapBuffer& get_global_gap() const { return global_gap_; }
    
    // Clear after reconstruction
    void clear_global_gap() {
        global_gap_.clear();
    }
    
private:
    int num_sub_hnsw_;
    int dimension_;
    double gap_ratio_;  // Target gap size as ratio of index capacity
    GlobalGapBuffer global_gap_;
};

}  // namespace reconstruction
}  // namespace dhnsw

