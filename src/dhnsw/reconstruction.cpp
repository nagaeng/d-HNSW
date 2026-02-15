// reconstruction.cpp - Implementation of reconstruction components

#include "reconstruction.hh"
#include "DistributedHnsw.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <sstream>
#include <cstring>

namespace dhnsw {
namespace reconstruction {

//==============================================================================
// LSHCache Implementation
//==============================================================================

LSHCache::LSHCache(int dimension, int num_hash_tables, int num_hash_bits)
    : dim_(dimension)
    , num_tables_(num_hash_tables)
    , num_bits_(num_hash_bits) {
    init_projections();
    hash_tables_.resize(num_tables_);
}

void LSHCache::init_projections() {
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    projection_matrices_.resize(num_tables_);
    for (int t = 0; t < num_tables_; ++t) {
        projection_matrices_[t].resize(num_bits_ * dim_);
        for (int i = 0; i < num_bits_ * dim_; ++i) {
            projection_matrices_[t][i] = dist(rng);
        }
    }
}

uint64_t LSHCache::compute_hash(const float* vector, int table_idx) const {
    uint64_t hash = 0;
    const float* proj = projection_matrices_[table_idx].data();
    
    for (int b = 0; b < num_bits_; ++b) {
        float dot = 0.0f;
        for (int d = 0; d < dim_; ++d) {
            dot += vector[d] * proj[b * dim_ + d];
        }
        if (dot > 0) {
            hash |= (1ULL << b);
        }
    }
    return hash;
}

void LSHCache::add(int64_t id, const float* vector) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t idx = vectors_.size() / dim_;
    
    // Store vector and id
    vectors_.insert(vectors_.end(), vector, vector + dim_);
    ids_.push_back(id);
    
    // Add to hash tables
    for (int t = 0; t < num_tables_; ++t) {
        uint64_t hash = compute_hash(vector, t);
        hash_tables_[t][hash].push_back(idx);
    }
}

void LSHCache::add_batch(const std::vector<int64_t>& ids, const float* vectors, int count) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t start_idx = vectors_.size() / dim_;
    
    // Store all vectors and ids
    vectors_.insert(vectors_.end(), vectors, vectors + count * dim_);
    ids_.insert(ids_.end(), ids.begin(), ids.end());
    
    // Add to hash tables
    for (int i = 0; i < count; ++i) {
        const float* vec = vectors + i * dim_;
        size_t idx = start_idx + i;
        
        for (int t = 0; t < num_tables_; ++t) {
            uint64_t hash = compute_hash(vec, t);
            hash_tables_[t][hash].push_back(idx);
        }
    }
}

std::vector<std::pair<int64_t, float>> LSHCache::search(const float* query, int k) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (vectors_.empty()) {
        return {};
    }
    
    // Collect candidate indices from all hash tables
    std::unordered_set<size_t> candidates;
    for (int t = 0; t < num_tables_; ++t) {
        uint64_t hash = compute_hash(query, t);
        auto it = hash_tables_[t].find(hash);
        if (it != hash_tables_[t].end()) {
            candidates.insert(it->second.begin(), it->second.end());
        }
    }
    
    // If no candidates from LSH, fall back to scanning all
    if (candidates.empty()) {
        size_t n = vectors_.size() / dim_;
        for (size_t i = 0; i < n; ++i) {
            candidates.insert(i);
        }
    }
    
    // Compute exact distances for candidates
    std::vector<std::pair<float, size_t>> distances;
    distances.reserve(candidates.size());
    
    for (size_t idx : candidates) {
        const float* vec = vectors_.data() + idx * dim_;
        float dist = 0.0f;
        for (int d = 0; d < dim_; ++d) {
            float diff = query[d] - vec[d];
            dist += diff * diff;
        }
        distances.emplace_back(dist, idx);
    }
    
    // Partial sort to get top k
    size_t result_k = std::min(static_cast<size_t>(k), distances.size());
    std::partial_sort(distances.begin(), distances.begin() + result_k, 
                     distances.end());
    
    // Build result
    std::vector<std::pair<int64_t, float>> results;
    results.reserve(result_k);
    for (size_t i = 0; i < result_k; ++i) {
        results.emplace_back(ids_[distances[i].second], 
                            std::sqrt(distances[i].first));
    }
    
    return results;
}

std::vector<float> LSHCache::get_all_vectors() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return vectors_;
}

std::vector<int64_t> LSHCache::get_all_ids() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return ids_;
}

void LSHCache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    vectors_.clear();
    ids_.clear();
    for (auto& table : hash_tables_) {
        table.clear();
    }
}

void LSHCache::reinitialize(int new_dim, int new_num_tables, int new_num_bits) {
    std::lock_guard<std::mutex> lock(mutex_);
    clear();
    dim_ = new_dim;
    num_tables_ = new_num_tables;
    num_bits_ = new_num_bits;
    init_projections();
    hash_tables_.resize(num_tables_);
}

std::vector<uint8_t> LSHCache::serialize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<uint8_t> data;
    
    // Header: dim, num_tables, num_bits
    size_t header_size = 3 * sizeof(int);
    size_t vectors_size = vectors_.size() * sizeof(float);
    size_t ids_size = ids_.size() * sizeof(int64_t);
    
    data.resize(header_size + sizeof(size_t) * 2 + vectors_size + ids_size);
    
    uint8_t* ptr = data.data();
    
    std::memcpy(ptr, &dim_, sizeof(int)); ptr += sizeof(int);
    std::memcpy(ptr, &num_tables_, sizeof(int)); ptr += sizeof(int);
    std::memcpy(ptr, &num_bits_, sizeof(int)); ptr += sizeof(int);
    
    size_t vec_count = vectors_.size();
    std::memcpy(ptr, &vec_count, sizeof(size_t)); ptr += sizeof(size_t);
    std::memcpy(ptr, vectors_.data(), vectors_size); ptr += vectors_size;
    
    size_t id_count = ids_.size();
    std::memcpy(ptr, &id_count, sizeof(size_t)); ptr += sizeof(size_t);
    std::memcpy(ptr, ids_.data(), ids_size);
    
    return data;
}

void LSHCache::deserialize_into(const std::vector<uint8_t>& data, LSHCache& cache) {
    const uint8_t* ptr = data.data();

    int dim, num_tables, num_bits;
    std::memcpy(&dim, ptr, sizeof(int)); ptr += sizeof(int);
    std::memcpy(&num_tables, ptr, sizeof(int)); ptr += sizeof(int);
    std::memcpy(&num_bits, ptr, sizeof(int)); ptr += sizeof(int);

    // Reinitialize the cache with new dimensions
    cache.reinitialize(dim, num_tables, num_bits);

    size_t vec_count;
    std::memcpy(&vec_count, ptr, sizeof(size_t)); ptr += sizeof(size_t);

    size_t id_count;
    const float* vec_ptr = reinterpret_cast<const float*>(ptr);
    ptr += vec_count * sizeof(float);

    std::memcpy(&id_count, ptr, sizeof(size_t)); ptr += sizeof(size_t);
    const int64_t* id_ptr = reinterpret_cast<const int64_t*>(ptr);

    // Reconstruct by adding vectors
    int n_vectors = vec_count / dim;
    for (int i = 0; i < n_vectors; ++i) {
        cache.add(id_ptr[i], vec_ptr + i * dim);
    }
}


//==============================================================================
// ThroughputLogger Implementation
//==============================================================================

ThroughputLogger::ThroughputLogger(const std::string& output_file, 
                                   const std::string& client_id)
    : output_file_(output_file)
    , client_id_(client_id)
    , last_sample_time_(Clock::now()) {
}

ThroughputLogger::~ThroughputLogger() {
    stop_logging();
    flush();
}

void ThroughputLogger::record_query(int count) {
    query_count_.fetch_add(count);
}

void ThroughputLogger::record_insert(int count) {
    insert_count_.fetch_add(count);
}

void ThroughputLogger::record_batch(int worker_id, uint64_t batch_start_ms, uint64_t batch_end_ms,
                                   int query_count, int insert_count, bool blocked) {
    std::lock_guard<std::mutex> lock(mutex_);
    batch_records_.push_back(BatchRecord{
        worker_id,
        batch_start_ms,
        batch_end_ms,
        query_count,
        insert_count,
        blocked,
        client_id_
    });
    
    // Also update counters for periodic sampling compatibility
    query_count_.fetch_add(query_count);
    insert_count_.fetch_add(insert_count);
}

void ThroughputLogger::set_blocked(bool blocked) {
    is_blocked_.store(blocked);
}

void ThroughputLogger::record_event(const std::string& event) {
    std::lock_guard<std::mutex> lock(mutex_);
    ThroughputSample sample = get_current_sample();
    sample.event = event;
    samples_.push_back(sample);
}

ThroughputSample ThroughputLogger::get_current_sample() const {
    auto now = Clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_sample_time_).count();

    if (duration == 0) duration = 1;  // Avoid division by zero

    int64_t queries = query_count_.load() - last_query_count_;
    int64_t inserts = insert_count_.load() - last_insert_count_;

    double qps = queries * 1000.0 / duration;
    double ips = inserts * 1000.0 / duration;

    return ThroughputSample{
        current_time_ms(),
        qps,
        ips,
        qps + ips,
        is_blocked_.load(),
        client_id_,
        ""  // event field (empty for regular samples)
    };
}

void ThroughputLogger::flush() {
    std::lock_guard<std::mutex> lock(mutex_);

    // Create directory if it doesn't exist
    std::filesystem::path file_path(output_file_);
    std::filesystem::path dir_path = file_path.parent_path();
    if (!dir_path.empty() && !std::filesystem::exists(dir_path)) {
        try {
            std::filesystem::create_directories(dir_path);
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Failed to create directory " << dir_path << ": " << e.what() << std::endl;
            return;
        }
    }

    // Write periodic samples
    std::ofstream file(output_file_, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Failed to open throughput log file: " << output_file_ << std::endl;
        return;
    }
    
    for (const auto& sample : samples_) {
        file << sample.timestamp_ms << ","
             << sample.client_id << ","
             << sample.qps << ","
             << sample.ips << ","
             << sample.total_ops << ","
             << (sample.is_blocked ? 1 : 0) << ","
             << sample.event << "\n";
    }
    
    file.close();
    samples_.clear();
    
    // Write batch records to a separate file for accurate throughput over time analysis
    if (!batch_records_.empty()) {
        std::string batch_file = output_file_;
        size_t dot_pos = batch_file.rfind('.');
        if (dot_pos != std::string::npos) {
            batch_file = batch_file.substr(0, dot_pos) + "_batches" + batch_file.substr(dot_pos);
        } else {
            batch_file += "_batches";
        }
        
        std::ofstream batch_out(batch_file, std::ios::app);
        if (!batch_out.is_open()) {
            std::cerr << "Failed to open batch log file: " << batch_file << std::endl;
            return;
        }
        
        // Write batch records with format:
        // client_id,worker_id,batch_start_ms,batch_end_ms,query_count,insert_count,blocked
        for (const auto& record : batch_records_) {
            batch_out << record.client_id << ","
                     << record.worker_id << ","
                     << record.batch_start_ms << ","
                     << record.batch_end_ms << ","
                     << record.query_count << ","
                     << record.insert_count << ","
                     << (record.blocked ? 1 : 0) << "\n";
        }
        
        batch_out.close();
        batch_records_.clear();
    }
}

void ThroughputLogger::start_logging(int interval_ms) {
    stop_logging_.store(false);
    logging_thread_ = std::thread(&ThroughputLogger::logging_worker, this, interval_ms);
}

void ThroughputLogger::stop_logging() {
    stop_logging_.store(true);
    if (logging_thread_.joinable()) {
        logging_thread_.join();
    }
}

void ThroughputLogger::logging_worker(int interval_ms) {
    while (!stop_logging_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        
        auto sample = get_current_sample();
        
        {
            std::lock_guard<std::mutex> lock(mutex_);
            samples_.push_back(sample);
            
            // Update baseline
            last_sample_time_ = Clock::now();
            last_query_count_ = query_count_.load();
            last_insert_count_ = insert_count_.load();
        }
        
        // Flush frequently to ensure data is captured even if worker crashes
        if (samples_.size() >= 10) {
            flush();
        }
    }
}


//==============================================================================
// ClientReconstructionHandler Implementation
//==============================================================================

ClientReconstructionHandler::ClientReconstructionHandler(
    const std::string& client_id, 
    bool is_master)
    : client_id_(client_id)
    , is_master_(is_master)
    , insert_cache_(128)  // Default dimension, will be resized as needed
{
}

ClientReconstructionHandler::~ClientReconstructionHandler() {
}

void ClientReconstructionHandler::enter_blocking_mode() {
    std::lock_guard<std::mutex> lock(mutex_);
    block_queries_.store(true);
}

void ClientReconstructionHandler::exit_blocking_mode() {
    std::lock_guard<std::mutex> lock(mutex_);
    block_queries_.store(false);
    cv_.notify_all();
}

bool ClientReconstructionHandler::wait_for_completion(int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    auto deadline = std::chrono::steady_clock::now() + 
                    std::chrono::milliseconds(timeout_ms);
    
    while (current_state_.phase != ReconstructionPhase::COMPLETED &&
           current_state_.phase != ReconstructionPhase::IDLE) {
        if (cv_.wait_until(lock, deadline) == std::cv_status::timeout) {
            return false;
        }
    }
    
    return true;
}

void ClientReconstructionHandler::handle_notification(const ReconstructionState& state) {
    std::lock_guard<std::mutex> lock(mutex_);
    current_state_ = state;
    
    if (state.phase == ReconstructionPhase::COMPLETED ||
        state.phase == ReconstructionPhase::IDLE) {
        block_queries_.store(false);
        cv_.notify_all();
        
        if (update_callback_) {
            update_callback_(state);
        }
    }
}

void ClientReconstructionHandler::sync_insert_cache(const std::vector<uint8_t>& cache_data) {
    if (is_master_) {
        return;  // Master doesn't need to sync, it owns the cache
    }

    std::lock_guard<std::mutex> lock(mutex_);
    LSHCache::deserialize_into(cache_data, insert_cache_);
}


//==============================================================================
// ServerReconstructionManager Implementation
//==============================================================================

ServerReconstructionManager::ServerReconstructionManager(
    int dimension,
    int num_sub_hnsw,
    int meta_hnsw_neighbors,
    int sub_hnsw_neighbors,
    int num_meta)
    : dim_(dimension)
    , num_sub_hnsw_(num_sub_hnsw)
    , meta_M_(meta_hnsw_neighbors)
    , sub_M_(sub_hnsw_neighbors)
    , num_meta_(num_meta) {
}

ServerReconstructionManager::~ServerReconstructionManager() {
    stop_reconstruction_.store(true);
    if (reconstruction_thread_.joinable()) {
        reconstruction_thread_.join();
    }
}

bool ServerReconstructionManager::needs_reconstruction(
    const std::vector<uint64_t>& overflow) const {
    return any_overflow_above_threshold(overflow, num_sub_hnsw_, overflow_threshold_);
}

uint64_t ServerReconstructionManager::start_reconstruction(
    const std::vector<float>& existing_data,
    const std::vector<float>& buffered_inserts,
    int existing_count,
    int insert_count) {
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (state_.phase != ReconstructionPhase::IDLE) {
        std::cerr << "Reconstruction already in progress" << std::endl;
        return 0;
    }
    
    // Generate new reconstruction ID
    state_.reconstruction_id = current_time_ms();
    state_.phase = ReconstructionPhase::TRIGGERED;
    state_.progress = 0.0;
    state_.start_time = Clock::now();
    state_.message = "Starting reconstruction";
    
    // Reset client acknowledgments
    {
        std::lock_guard<std::mutex> clients_lock(clients_mutex_);
        for (auto& kv : clients_) {
            kv.second.acknowledged = false;
        }
    }
    
    // Start background reconstruction
    if (reconstruction_thread_.joinable()) {
        reconstruction_thread_.join();
    }
    
    stop_reconstruction_.store(false);
    reconstruction_thread_ = std::thread(
        &ServerReconstructionManager::reconstruction_worker,
        this,
        existing_data,
        buffered_inserts,
        existing_count,
        insert_count
    );
    
    return state_.reconstruction_id;
}

ReconstructionState ServerReconstructionManager::get_status() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return state_;
}

bool ServerReconstructionManager::get_rebuilt_data(
    std::vector<uint8_t>& meta_hnsw,
    std::vector<uint64_t>& offset_subhnsw,
    std::vector<uint64_t>& offset_para,
    std::vector<uint64_t>& overflow,
    std::vector<std::vector<int64_t>>& mapping,
    std::vector<uint8_t>& serialized_data) {
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    if (state_.phase != ReconstructionPhase::COMPLETED &&
        state_.phase != ReconstructionPhase::WAITING_FOR_ACKS) {
        return false;
    }
    
    meta_hnsw = state_.new_meta_hnsw;
    offset_subhnsw = state_.new_offset_subhnsw;
    offset_para = state_.new_offset_para;
    overflow = state_.new_overflow;
    mapping = state_.new_mapping;
    serialized_data = rebuilt_serialized_data_;
    
    return true;
}

void ServerReconstructionManager::register_client(
    const std::string& client_id, 
    const std::string& address, 
    bool is_master) {
    
    std::lock_guard<std::mutex> lock(clients_mutex_);
    clients_[client_id] = {address, is_master, false};
    
    std::cout << "Registered client: " << client_id 
              << " at " << address 
              << " (master=" << is_master << ")" << std::endl;
}

void ServerReconstructionManager::unregister_client(const std::string& client_id) {
    std::lock_guard<std::mutex> lock(clients_mutex_);
    clients_.erase(client_id);
}

bool ServerReconstructionManager::acknowledge_completion(
    const std::string& client_id, 
    uint64_t reconstruction_id) {
    
    std::lock_guard<std::mutex> clients_lock(clients_mutex_);
    std::lock_guard<std::mutex> state_lock(state_mutex_);
    
    if (state_.reconstruction_id != reconstruction_id) {
        return false;
    }
    
    auto it = clients_.find(client_id);
    if (it != clients_.end()) {
        it->second.acknowledged = true;
    }
    
    // Check if all acknowledged
    bool all_acked = true;
    for (const auto& kv : clients_) {
        if (!kv.second.acknowledged) {
            all_acked = false;
            break;
        }
    }
    
    if (all_acked && state_.phase == ReconstructionPhase::WAITING_FOR_ACKS) {
        state_.phase = ReconstructionPhase::COMPLETED;
        state_.end_time = Clock::now();
        state_.progress = 100.0;
        state_.message = "Reconstruction completed, all clients acknowledged";
        
        // Reset to IDLE after short delay
        std::thread([this]() {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (state_.phase == ReconstructionPhase::COMPLETED) {
                state_.phase = ReconstructionPhase::IDLE;
            }
        }).detach();
    }
    
    return all_acked;
}

bool ServerReconstructionManager::all_clients_acknowledged() const {
    std::lock_guard<std::mutex> lock(clients_mutex_);

    for (const auto& kv : clients_) {
        if (!kv.second.acknowledged) {
            return false;
        }
    }
    return true;
}

void ServerReconstructionManager::force_complete() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.phase = ReconstructionPhase::COMPLETED;
    state_.progress = 100.0;
    state_.end_time = Clock::now();
    state_.message = "Reconstruction completed";

    // Transition to IDLE after a brief delay
    std::thread([this]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (state_.phase == ReconstructionPhase::COMPLETED) {
            state_.phase = ReconstructionPhase::IDLE;
            state_.message = "Idle";
        }
    }).detach();
}

void ServerReconstructionManager::force_idle() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    state_.phase = ReconstructionPhase::IDLE;
    state_.progress = 0.0;
    state_.message = "Idle (reset after error)";
}

void ServerReconstructionManager::reconstruction_worker(
    std::vector<float> existing_data,
    std::vector<float> buffered_inserts,
    int existing_count,
    int insert_count) {
    
    try {
        // Phase 1: Building index
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state_.phase = ReconstructionPhase::BUILDING_INDEX;
            state_.progress = 10.0;
            state_.message = "Building new index";
        }
        
        // Combine existing data with buffered inserts
        std::vector<float> combined_data;
        combined_data.reserve(existing_data.size() + buffered_inserts.size());
        combined_data.insert(combined_data.end(), 
                            existing_data.begin(), existing_data.end());
        combined_data.insert(combined_data.end(), 
                            buffered_inserts.begin(), buffered_inserts.end());
        
        int total_count = existing_count + insert_count;
        
        // Create new DistributedHnsw
        DistributedHnsw new_dhnsw(dim_, num_sub_hnsw_, meta_M_, sub_M_, num_meta_);
        
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state_.progress = 30.0;
            state_.message = "Building distributed HNSW structure";
        }
        
        // Build the new index
        new_dhnsw.build(combined_data, num_meta_);
        
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state_.progress = 60.0;
            state_.message = "Serializing new index";
        }
        
        // Serialize the new index
        std::vector<uint64_t> new_offset_subhnsw;
        std::vector<uint64_t> new_offset_para;
        std::vector<uint64_t> new_overflow;
        
        rebuilt_serialized_data_ = new_dhnsw.serialize_with_record_with_in_out_gap(
            new_offset_subhnsw, new_offset_para, new_overflow);
        
        std::vector<uint8_t> new_meta_hnsw = new_dhnsw.serialize_meta_hnsw();
        std::vector<std::vector<int64_t>> new_mapping_int64;
        auto mapping = new_dhnsw.get_mapping();
        for (const auto& m : mapping) {
            new_mapping_int64.push_back(std::vector<int64_t>(m.begin(), m.end()));
        }
        
        // Phase 2: Copying data
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state_.phase = ReconstructionPhase::COPYING_DATA;
            state_.progress = 80.0;
            state_.message = "Copying data to RDMA memory";
            
            state_.new_meta_hnsw = new_meta_hnsw;
            state_.new_offset_subhnsw = new_offset_subhnsw;
            state_.new_offset_para = new_offset_para;
            state_.new_overflow = new_overflow;
            state_.new_mapping = new_mapping_int64;
        }
        
        // Phase 3: Notifying clients
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state_.phase = ReconstructionPhase::NOTIFYING_CLIENTS;
            state_.progress = 90.0;
            state_.message = "Notifying clients of completion";
        }
        
        // Phase 4: Waiting for acknowledgments
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            state_.phase = ReconstructionPhase::WAITING_FOR_ACKS;
            state_.progress = 95.0;
            state_.message = "Waiting for client acknowledgments";
        }
        
        std::cout << "Reconstruction complete, waiting for client acknowledgments" 
                  << std::endl;
        
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        state_.phase = ReconstructionPhase::IDLE;
        state_.progress = 0.0;
        state_.message = std::string("Reconstruction failed: ") + e.what();
        std::cerr << "Reconstruction failed: " << e.what() << std::endl;
    }
}

}  // namespace reconstruction
}  // namespace dhnsw

