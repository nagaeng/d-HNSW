// search_server_reconstruction.cc
// Memory server with reconstruction support for distributed HNSW

#include <gflags/gflags.h>
#include <grpcpp/grpcpp.h>
#include "../generated/dhnsw.grpc.pb.h"

#include "../../deps/rlib/core/lib.hh"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <unordered_set>

#include "../dhnsw/DistributedHnsw.h"
#include "../dhnsw/reconstruction.hh"
#include "../util/read_dataset.h"

DEFINE_int32(port, 50051, "Port for the gRPC server to listen on.");
DEFINE_int32(rdma_port, 8888, "Port for the RDMA control channel.");
DEFINE_int32(use_nic_idx, 3, "Which NIC to create QP");
DEFINE_int32(reg_nic_name, 73, "The name to register an opened NIC at rctrl.");
DEFINE_int32(reg_mem_name, 73, "The name to register an MR at rctrl.");
DEFINE_string(dataset_path, "../datasets/sift/sift_base.fvecs", "Path to the dataset.");
DEFINE_string(server_ip, "0.0.0.0", "IP address for the gRPC server to bind.");
DEFINE_double(overflow_threshold, 1, "Threshold for triggering reconstruction (0.0-1.0)");
DEFINE_int32(dim, 128, "Vector dimension");
DEFINE_int32(num_sub_hnsw, 160, "Number of sub-HNSW indices");
DEFINE_int32(meta_hnsw_neighbors, 32, "Meta HNSW neighbors");
DEFINE_int32(sub_hnsw_neighbors, 48, "Sub HNSW neighbors");
DEFINE_int32(num_meta, 5000, "Number of meta vectors");

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using dhnsw::DhnswService;
using dhnsw::Empty;
using dhnsw::MetaHnswResponse;
using dhnsw::MappingResponse;
using dhnsw::MappingEntry;
using dhnsw::OverflowResponse;
using dhnsw::Offset_SubHnswResponse;
using dhnsw::Offset_ParaResponse;
using namespace rdmaio;
using namespace rdmaio::rmem;
using namespace rdmaio::qp;
using namespace dhnsw::reconstruction;

// Global state for reconstruction coordination (defined before service class that uses it)
struct ServerState {
    std::mutex data_mutex;

    // Current index data
    std::vector<uint8_t> serialized_meta_hnsw;
    std::vector<size_t> offset_sub_hnsw;
    std::vector<size_t> offset_para;
    std::vector<size_t> overflow;
    std::vector<std::vector<dhnsw_idx_t>> mapping;
    std::vector<uint8_t> serialized_data;

    // RDMA memory
    Arc<RMem> mr_memory;
    uint8_t* rdma_buffer = nullptr;
    size_t rdma_buffer_size = 0;

    // Epoch-based buffer manager for safe reconstruction
    std::unique_ptr<EpochBufferManager> epoch_buffer_manager;

    // Reconstruction manager
    std::unique_ptr<ServerReconstructionManager> reconstruction_manager;

    // For double buffering during reconstruction
    std::vector<uint8_t> new_serialized_data;
    uint64_t current_rdma_offset = 0;

    // Registered clients for notification
    std::unordered_map<std::string, std::string> registered_clients;
    std::mutex clients_mutex;

    // Reconstruction state
    std::atomic<bool> reconstruction_in_progress{false};
    std::atomic<uint64_t> current_reconstruction_id{0};
    std::unordered_set<std::string> acknowledged_clients;
    std::mutex ack_mutex;
    std::condition_variable ack_cv;

    // Epoch-based coordination
    std::atomic<uint64_t> active_epoch{0};
    std::unordered_map<uint64_t, uint64_t> epoch_rdma_offsets;  // epoch -> rdma_offset
    std::unordered_map<uint64_t, std::unordered_set<std::string>> epoch_acks;  // epoch -> clients that acked
    std::mutex epoch_mutex;

    // Per-epoch metadata snapshots for consistent reads during reconstruction
    struct EpochMetadata {
        std::vector<uint8_t> serialized_meta_hnsw;
        std::vector<size_t> offset_sub_hnsw;
        std::vector<size_t> offset_para;
        std::vector<size_t> overflow;
        std::vector<std::vector<dhnsw_idx_t>> mapping;
        uint64_t rdma_base_offset;
    };
    std::unordered_map<uint64_t, EpochMetadata> epoch_metadata;
    std::mutex epoch_metadata_mutex;

    // Insert cache storage per epoch (for workers to fetch)
    struct EpochInsertCache {
        std::vector<float> vectors;
        std::vector<int64_t> ids;
        int dimension;
    };
    std::unordered_map<uint64_t, EpochInsertCache> insert_cache_per_epoch;
    std::mutex insert_cache_mutex;

    // Old epoch buffers pending reclamation
    std::vector<std::pair<uint64_t, uint64_t>> pending_reclaim;  // (epoch, offset)
    std::mutex reclaim_mutex;

    // Store base vectors for reconstruction (raw float data)
    std::vector<float> base_vectors;
    int base_vector_count = 0;
    int vector_dim = 0;

    // Tiered insert buffer for reduced reconstruction frequency
    std::unique_ptr<TieredInsertBuffer> tiered_insert_buffer;
};

// Reconstruction coordination thread (defined before service that uses it)
class ReconstructionCoordinator {
public:
    ReconstructionCoordinator(ServerState* state, RCtrl* rdma_ctrl)
        : state_(state)
        , rdma_ctrl_(rdma_ctrl)
        , stop_(false) {}

    ~ReconstructionCoordinator() {
        stop();
    }

    void start() {
        coordinator_thread_ = std::thread(&ReconstructionCoordinator::run, this);
    }

    void stop() {
        stop_.store(true);
        cv_.notify_all();
        if (coordinator_thread_.joinable()) {
            coordinator_thread_.join();
        }
    }

    // Called by insert client to trigger reconstruction
    bool trigger_reconstruction(
        const std::vector<float>& buffered_vectors,
        int vector_count) {

        if (state_->reconstruction_in_progress.load()) {
            return false;  // Already in progress
        }

        std::lock_guard<std::mutex> lock(trigger_mutex_);

        buffered_vectors_ = buffered_vectors;
        buffered_count_ = vector_count;
        trigger_reconstruction_.store(true);
        cv_.notify_one();

        return true;
    }

    // Get current reconstruction status
    ReconstructionState get_status() const {
        return state_->reconstruction_manager->get_status();
    }

private:
    void run() {
        while (!stop_.load()) {
            std::unique_lock<std::mutex> lock(trigger_mutex_);
            cv_.wait(lock, [this]() {
                return trigger_reconstruction_.load() || stop_.load();
            });

            if (stop_.load()) break;

            if (trigger_reconstruction_.load()) {
                trigger_reconstruction_.store(false);
                perform_reconstruction();
            }
        }
    }

    void perform_reconstruction() {
        state_->reconstruction_in_progress.store(true);
        auto start_time = std::chrono::high_resolution_clock::now();

        uint64_t old_epoch = state_->active_epoch.load();
        uint64_t new_epoch = old_epoch + 1;

        try {
            // Step 1: Store insert cache for this epoch (workers will fetch)
            {
                std::lock_guard<std::mutex> cache_lock(state_->insert_cache_mutex);
                ServerState::EpochInsertCache cache;
                cache.vectors = buffered_vectors_;
                cache.dimension = (buffered_count_ > 0 && buffered_vectors_.size() > 0)
                    ? static_cast<int>(buffered_vectors_.size() / buffered_count_) : 0;
                for (int i = 0; i < buffered_count_; ++i) {
                    cache.ids.push_back(static_cast<int64_t>(i));
                }
                state_->insert_cache_per_epoch[new_epoch] = std::move(cache);
            }

            // Step 2: Use stored base vectors for reconstruction
            std::vector<float> existing_data;
            int existing_count = 0;
            {
                std::lock_guard<std::mutex> lock(state_->data_mutex);
                existing_data = state_->base_vectors;
                existing_count = state_->base_vector_count;
            }

            auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            std::cout << "{\"event\":\"TRIGGER_RECONSTRUCTION\""
                      << ",\"timestamp_ms\":" << now_ms
                      << ",\"old_epoch\":" << old_epoch
                      << ",\"new_epoch\":" << new_epoch
                      << ",\"existing_vectors\":" << existing_count
                      << ",\"buffered_vectors\":" << buffered_count_
                      << "}" << std::endl;

            // Step 3: Start background reconstruction with existing + buffered vectors
            uint64_t reconstruction_id = state_->reconstruction_manager->start_reconstruction(
                existing_data, buffered_vectors_, existing_count, buffered_count_);

            state_->current_reconstruction_id.store(reconstruction_id);

            // Step 4: Wait for reconstruction to complete
            while (true) {
                auto status = state_->reconstruction_manager->get_status();
                if (status.phase == ReconstructionPhase::WAITING_FOR_ACKS ||
                    status.phase == ReconstructionPhase::COMPLETED ||
                    status.phase == ReconstructionPhase::IDLE) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            // Step 5: Get rebuilt data
            std::vector<uint8_t> new_meta_hnsw;
            std::vector<uint64_t> new_offset_subhnsw;
            std::vector<uint64_t> new_offset_para;
            std::vector<uint64_t> new_overflow;
            std::vector<std::vector<int64_t>> new_mapping;
            std::vector<uint8_t> new_serialized_data;

            bool success = state_->reconstruction_manager->get_rebuilt_data(
                new_meta_hnsw, new_offset_subhnsw, new_offset_para,
                new_overflow, new_mapping, new_serialized_data);

            if (!success) {
                std::cerr << "Failed to get rebuilt data" << std::endl;
                state_->reconstruction_in_progress.store(false);
                return;
            }

            // Step 6: SAFE DOUBLE-BUFFERING using EpochBufferManager
            // This ensures we don't overwrite memory that readers are still using
            size_t new_data_size = new_serialized_data.size();
            uint64_t new_offset = 0;
            
            if (state_->epoch_buffer_manager) {
                // Use epoch manager for safe buffer allocation
                new_offset = state_->epoch_buffer_manager->prepare_new_epoch(new_data_size);
                
                std::cout << "{\"event\":\"BUFFER_ALLOCATED\""
                          << ",\"new_offset\":" << new_offset
                          << ",\"new_data_size\":" << new_data_size
                          << ",\"old_epoch_readers\":" << state_->epoch_buffer_manager->get_active_readers(old_epoch)
                          << "}" << std::endl;
            } else {
                // Fallback to old behavior if epoch manager not initialized
                new_offset = state_->current_rdma_offset + state_->serialized_data.size();
                if (new_offset + new_data_size > state_->rdma_buffer_size) {
                    new_offset = 0;
                }
            }

            // Copy new data to RDMA buffer
            std::memcpy(state_->rdma_buffer + new_offset,
                       new_serialized_data.data(),
                       new_data_size);

            // Step 7: Store epoch metadata snapshot BEFORE switching epoch
            // This allows clients reading old epoch to get consistent metadata
            {
                std::lock_guard<std::mutex> meta_lock(state_->epoch_metadata_mutex);
                ServerState::EpochMetadata new_meta;
                new_meta.serialized_meta_hnsw = new_meta_hnsw;
                // Store relative offsets - client will add rdma_base_offset when reading
                new_meta.offset_sub_hnsw = std::vector<size_t>(
                    new_offset_subhnsw.begin(), new_offset_subhnsw.end());
                new_meta.offset_para = std::vector<size_t>(
                    new_offset_para.begin(), new_offset_para.end());
                new_meta.overflow = std::vector<size_t>(
                    new_overflow.begin(), new_overflow.end());
                for (const auto& m : new_mapping) {
                    new_meta.mapping.push_back(std::vector<dhnsw_idx_t>(m.begin(), m.end()));
                }
                new_meta.rdma_base_offset = new_offset;
                state_->epoch_metadata[new_epoch] = std::move(new_meta);
                
                // Keep only last 3 epochs of metadata
                if (state_->epoch_metadata.size() > 3) {
                    auto oldest_epoch = new_epoch > 2 ? new_epoch - 2 : 0;
                    for (auto it = state_->epoch_metadata.begin(); it != state_->epoch_metadata.end(); ) {
                        if (it->first < oldest_epoch) {
                            it = state_->epoch_metadata.erase(it);
                        } else {
                            ++it;
                        }
                    }
                }
            }

            // Step 8: Track old epoch buffer for reclamation
            {
                std::lock_guard<std::mutex> reclaim_lock(state_->reclaim_mutex);
                state_->pending_reclaim.push_back({old_epoch, state_->current_rdma_offset});
            }

            // Step 9: Atomically update metadata and epoch
            {
                std::lock_guard<std::mutex> lock(state_->data_mutex);

                state_->serialized_meta_hnsw = new_meta_hnsw;
                // Store relative offsets - client will add rdma_base_offset when reading
                state_->offset_sub_hnsw = std::vector<size_t>(
                    new_offset_subhnsw.begin(), new_offset_subhnsw.end());
                state_->offset_para = std::vector<size_t>(
                    new_offset_para.begin(), new_offset_para.end());
                state_->overflow = std::vector<size_t>(
                    new_overflow.begin(), new_overflow.end());

                state_->mapping.clear();
                for (const auto& m : new_mapping) {
                    state_->mapping.push_back(std::vector<dhnsw_idx_t>(m.begin(), m.end()));
                }

                state_->serialized_data = new_serialized_data;
                state_->current_rdma_offset = new_offset;

                // Update base_vectors to include newly inserted vectors
                state_->base_vectors.insert(state_->base_vectors.end(),
                                           buffered_vectors_.begin(),
                                           buffered_vectors_.end());
                state_->base_vector_count += buffered_count_;
            }

            // Step 10: Commit the new epoch - this makes it visible to readers
            if (state_->epoch_buffer_manager) {
                state_->epoch_buffer_manager->commit_new_epoch();
            }

            // Step 11: Update epoch atomically
            {
                std::lock_guard<std::mutex> epoch_lock(state_->epoch_mutex);
                state_->epoch_rdma_offsets[new_epoch] = new_offset;
                state_->active_epoch.store(new_epoch);
            }

            // Step 12: Wait for old epoch readers to quiesce before allowing another reconstruction
            if (state_->epoch_buffer_manager) {
                bool quiesced = state_->epoch_buffer_manager->wait_for_old_readers_quiesce(5);
                if (!quiesced) {
                    std::cerr << "Warning: Old epoch readers did not quiesce within timeout, "
                              << "remaining: " << state_->epoch_buffer_manager->get_active_readers(old_epoch)
                              << std::endl;
                }
            }

            // Step 13: Wait for clients to acknowledge
            // CRITICAL: Wait indefinitely for ACKs - do NOT timeout and reclaim
            // Clients must complete their init() and ACK before we can safely proceed
            // Premature reclamation causes use-after-free crashes
            bool all_acked = false;
            {
                std::unique_lock<std::mutex> ack_lock(state_->ack_mutex);
                state_->acknowledged_clients.clear();

                size_t required_acks = state_->registered_clients.size();
                if (required_acks > 0) {
                    std::cout << "Reconstruction complete, waiting indefinitely for " << required_acks 
                              << " client acknowledgments (strict quiescence)..." << std::endl;

                    // Wait indefinitely - no timeout
                    // This is safe because:
                    // 1. Clients will eventually detect epoch change and ACK
                    // 2. If a client crashes, it will be detected by other means
                    // 3. Premature timeout causes crashes which is worse than waiting
                    while (state_->acknowledged_clients.size() < required_acks) {
                        // Use 30-second intervals to log progress, but keep waiting
                        if (state_->ack_cv.wait_for(ack_lock, std::chrono::seconds(30)) ==
                            std::cv_status::timeout) {
                            std::cout << "Still waiting for client acknowledgments ("
                                      << state_->acknowledged_clients.size() << "/"
                                      << required_acks << " received)..." << std::endl;
                            // Don't break - keep waiting
                        }
                    }
                    all_acked = true;  // Only reach here when all clients acked
                } else {
                    all_acked = true;
                }
            }

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();

            auto end_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            std::cout << "{\"event\":\"RECONSTRUCTION_COMPLETE\""
                      << ",\"timestamp_ms\":" << end_ms
                      << ",\"epoch\":" << new_epoch
                      << ",\"duration_ms\":" << duration
                      << ",\"new_data_size\":" << new_data_size
                      << ",\"all_clients_acked\":" << (all_acked ? "true" : "false")
                      << "}" << std::endl;

            // Step 14: Update ServerReconstructionManager state to COMPLETED/IDLE
            if (state_->reconstruction_manager) {
                state_->reconstruction_manager->force_complete();
            }

        } catch (const std::exception& e) {
            std::cerr << "Reconstruction failed: " << e.what() << std::endl;
            if (state_->reconstruction_manager) {
                state_->reconstruction_manager->force_idle();
            }
        }

        state_->reconstruction_in_progress.store(false);
    }

    ServerState* state_;
    RCtrl* rdma_ctrl_;

    std::thread coordinator_thread_;
    std::atomic<bool> stop_;
    std::mutex trigger_mutex_;
    std::condition_variable cv_;

    std::atomic<bool> trigger_reconstruction_{false};
    std::vector<float> buffered_vectors_;
    int buffered_count_{0};
};

// Extended service implementation with reconstruction support
class DhnswServiceWithReconstruction final : public DhnswService::Service {
public:
    DhnswServiceWithReconstruction(
        std::vector<uint8_t>* serialized_meta_hnsw,
        std::vector<size_t>* offset_sub_hnsw,
        std::vector<size_t>* offset_para,
        std::vector<size_t>* overflow,
        std::vector<std::vector<dhnsw_idx_t>>* mapping,
        ServerReconstructionManager* reconstruction_manager,
        std::mutex* data_mutex,
        std::atomic<bool>* reconstruction_in_progress,
        std::atomic<uint64_t>* current_reconstruction_id,
        std::unordered_set<std::string>* acknowledged_clients,
        std::mutex* ack_mutex,
        std::condition_variable* ack_cv,
        std::unordered_map<std::string, std::string>* registered_clients,
        std::mutex* clients_mutex)
        : serialized_meta_hnsw_(serialized_meta_hnsw)
        , offset_sub_hnsw_(offset_sub_hnsw)
        , offset_para_(offset_para)
        , overflow_(overflow)
        , mapping_(mapping)
        , reconstruction_manager_(reconstruction_manager)
        , data_mutex_(data_mutex)
        , reconstruction_in_progress_(reconstruction_in_progress)
        , current_reconstruction_id_(current_reconstruction_id)
        , acknowledged_clients_(acknowledged_clients)
        , ack_mutex_(ack_mutex)
        , ack_cv_(ack_cv)
        , registered_clients_(registered_clients)
        , clients_mutex_(clients_mutex) {}

    void set_coordinator(ReconstructionCoordinator* coordinator) {
        coordinator_ = coordinator;
    }

    Status GetMetaHnsw(ServerContext* context, const Empty* request,
                       MetaHnswResponse* response) override {
        std::lock_guard<std::mutex> lock(*data_mutex_);
        response->set_serialized_meta_hnsw(
            std::string(serialized_meta_hnsw_->begin(), serialized_meta_hnsw_->end()));
        return Status::OK;
    }

    Status GetOffset_SubHnsw(ServerContext* context, const Empty* request,
                     Offset_SubHnswResponse* response) override {
        std::lock_guard<std::mutex> lock(*data_mutex_);
        for (size_t off : *offset_sub_hnsw_) {
            response->add_offsets_subhnsw(off);
        }
        return Status::OK;
    }

    Status GetOffset_Para(ServerContext* context, const Empty* request,
                     Offset_ParaResponse* response) override {
        std::lock_guard<std::mutex> lock(*data_mutex_);
        for (size_t off : *offset_para_) {
            response->add_offsets_para(off);
        }
        return Status::OK;
    }

    Status GetMapping(ServerContext* context, const Empty* request,
                      MappingResponse* response) override {
        std::lock_guard<std::mutex> lock(*data_mutex_);
        for (size_t i = 0; i < mapping_->size(); ++i) {
            MappingEntry* entry = response->add_entries();
            entry->set_sub_index(static_cast<uint32_t>(i));
            for (dhnsw_idx_t idx : (*mapping_)[i]) {
                entry->add_mapping(idx);
            }
        }
        return Status::OK;
    }

    Status GetOverflow(ServerContext* context, const Empty* request,
                     OverflowResponse* response) override {
        std::lock_guard<std::mutex> lock(*data_mutex_);
        for (size_t off : *overflow_) {
            response->add_overflow(off);
        }
        return Status::OK;
    }

    // Reconstruction RPC handlers
    Status TriggerReconstruction(ServerContext* context,
                                 const dhnsw::TriggerReconstructionRequest* request,
                                 dhnsw::TriggerReconstructionResponse* response) override {
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        std::cout << "{\"event\":\"TRIGGER_RECONSTRUCTION_RPC\""
                  << ",\"timestamp_ms\":" << now_ms
                  << ",\"client_id\":\"" << request->client_id() << "\""
                  << ",\"vector_count\":" << request->vector_count() << "}" << std::endl;

        // Check if reconstruction is already in progress
        if (reconstruction_in_progress_->load()) {
            response->set_success(false);
            response->set_reconstruction_id(current_reconstruction_id_->load());
            response->set_message("Reconstruction already in progress");
            return Status::OK;
        }

        // Register the client
        {
            std::lock_guard<std::mutex> lock(*clients_mutex_);
            (*registered_clients_)[request->client_id()] = request->client_id();
        }

        // Extract buffered vectors
        std::vector<float> buffered_vectors(request->buffered_vectors().begin(),
                                            request->buffered_vectors().end());

        // Trigger reconstruction via coordinator
        if (coordinator_ != nullptr) {
            bool triggered = coordinator_->trigger_reconstruction(
                buffered_vectors, request->vector_count());

            if (triggered) {
                // Generate new reconstruction ID
                uint64_t new_id = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                current_reconstruction_id_->store(new_id);

                response->set_success(true);
                response->set_reconstruction_id(new_id);
                response->set_message("Reconstruction started");

                std::cout << "{\"event\":\"RECONSTRUCTION_STARTED\""
                          << ",\"timestamp_ms\":" << now_ms
                          << ",\"reconstruction_id\":" << new_id << "}" << std::endl;
            } else {
                response->set_success(false);
                response->set_reconstruction_id(0);
                response->set_message("Failed to trigger reconstruction");
            }
        } else {
            response->set_success(false);
            response->set_reconstruction_id(0);
            response->set_message("Coordinator not initialized");
        }

        return Status::OK;
    }

    Status GetReconstructionStatus(ServerContext* context,
                                   const Empty* request,
                                   dhnsw::ReconstructionStatusResponse* response) override {
        auto status = reconstruction_manager_->get_status();

        response->set_reconstruction_id(current_reconstruction_id_->load());
        response->set_phase(static_cast<int>(status.phase));
        response->set_progress(status.progress);
        response->set_message(status.message);

        return Status::OK;
    }

    Status AcknowledgeReconstruction(ServerContext* context,
                                     const dhnsw::AckReconstructionRequest* request,
                                     dhnsw::AckReconstructionResponse* response) override {
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        uint64_t acked_epoch = request->epoch();

        std::cout << "{\"event\":\"ACK_EPOCH\""
                  << ",\"timestamp_ms\":" << now_ms
                  << ",\"client_id\":\"" << request->client_id() << "\""
                  << ",\"epoch\":" << acked_epoch
                  << ",\"reconstruction_id\":" << request->reconstruction_id() << "}" << std::endl;

        {
            std::lock_guard<std::mutex> lock(*ack_mutex_);
            acknowledged_clients_->insert(request->client_id());

            // Track epoch-specific acknowledgments
            if (state_) {
                std::lock_guard<std::mutex> epoch_lock(state_->epoch_mutex);
                state_->epoch_acks[acked_epoch].insert(request->client_id());

                // Check if all clients acked this epoch - then we can reclaim old buffer
                if (state_->epoch_acks[acked_epoch].size() >= registered_clients_->size()) {
                    try_reclaim_old_epochs(acked_epoch);
                }
            }

            // Check if all registered clients have acknowledged
            bool all_acked = (acknowledged_clients_->size() >= registered_clients_->size());
            response->set_all_acknowledged(all_acked);
            response->set_remaining_clients(registered_clients_->size() - acknowledged_clients_->size());

            if (all_acked) {
                std::cout << "{\"event\":\"ALL_CLIENTS_ACKNOWLEDGED\""
                          << ",\"timestamp_ms\":" << now_ms
                          << ",\"reconstruction_id\":" << request->reconstruction_id()
                          << ",\"epoch\":" << acked_epoch << "}" << std::endl;
            }
        }

        ack_cv_->notify_all();

        return Status::OK;
    }

    Status GetEpochInfo(ServerContext* context,
                        const Empty* request,
                        dhnsw::EpochInfoResponse* response) override {
        uint64_t epoch = 0;
        uint64_t rdma_offset = 0;
        int active_readers = 0;

        if (state_) {
            epoch = state_->active_epoch.load();
            
            // Use epoch buffer manager if available for accurate offset
            if (state_->epoch_buffer_manager) {
                rdma_offset = state_->epoch_buffer_manager->get_current_offset();
                active_readers = state_->epoch_buffer_manager->get_active_readers(epoch);
            } else {
                std::lock_guard<std::mutex> lock(state_->epoch_mutex);
                auto it = state_->epoch_rdma_offsets.find(epoch);
                if (it != state_->epoch_rdma_offsets.end()) {
                    rdma_offset = it->second;
                } else {
                    rdma_offset = state_->current_rdma_offset;
                }
            }
        }

        response->set_epoch(epoch);
        response->set_rdma_offset(rdma_offset);
        response->set_reconstruction_id(current_reconstruction_id_->load());
        response->set_reconstruction_in_progress(reconstruction_in_progress_->load());

        return Status::OK;
    }

    // New RPC: Acquire epoch read reference (for safe RDMA reads during reconstruction)
    // Returns ALL metadata for the acquired epoch atomically - no separate RPCs needed
    Status AcquireEpochRead(ServerContext* context,
                            const dhnsw::AcquireEpochRequest* request,
                            dhnsw::AcquireEpochResponse* response) override {
        if (state_ && state_->epoch_buffer_manager) {
            auto [epoch, offset] = state_->epoch_buffer_manager->acquire_read();
            response->set_epoch(epoch);
            response->set_rdma_base_offset(offset);
            response->set_success(true);
            
            // Return ALL metadata for this epoch atomically
            std::lock_guard<std::mutex> meta_lock(state_->epoch_metadata_mutex);
            auto it = state_->epoch_metadata.find(epoch);
            if (it != state_->epoch_metadata.end()) {
                response->set_has_metadata(true);
                
                // Offsets
                for (size_t off : it->second.offset_sub_hnsw) {
                    response->add_offset_subhnsw(off);
                }
                for (size_t off : it->second.offset_para) {
                    response->add_offset_para(off);
                }
                for (size_t off : it->second.overflow) {
                    response->add_overflow(off);
                }
                
                // Serialized meta HNSW
                response->set_serialized_meta_hnsw(
                    std::string(it->second.serialized_meta_hnsw.begin(), 
                               it->second.serialized_meta_hnsw.end()));
                
                // Mapping
                for (const auto& m : it->second.mapping) {
                    auto* entry = response->add_mapping();
                    for (dhnsw_idx_t idx : m) {
                        entry->add_mapping(idx);
                    }
                }
            } else {
                response->set_has_metadata(false);
            }
        } else {
            // Fallback behavior
            response->set_epoch(state_ ? state_->active_epoch.load() : 0);
            response->set_rdma_base_offset(state_ ? state_->current_rdma_offset : 0);
            response->set_success(true);
            response->set_has_metadata(false);
        }
        return Status::OK;
    }

    // New RPC: Release epoch read reference
    Status ReleaseEpochRead(ServerContext* context,
                            const dhnsw::ReleaseEpochRequest* request,
                            dhnsw::ReleaseEpochResponse* response) override {
        if (state_ && state_->epoch_buffer_manager) {
            state_->epoch_buffer_manager->release_read(request->epoch());
            response->set_success(true);
        } else {
            response->set_success(true);  // No-op if not using epoch manager
        }
        return Status::OK;
    }

    Status GetInsertCache(ServerContext* context,
                          const dhnsw::GetInsertCacheRequest* request,
                          dhnsw::InsertCacheResponse* response) override {
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        uint64_t requested_epoch = request->epoch();
        std::string client_id = request->client_id();

        bool has_cache = false;
        int vector_count = 0;
        int dimension = 0;

        if (state_) {
            std::lock_guard<std::mutex> lock(state_->insert_cache_mutex);
            auto it = state_->insert_cache_per_epoch.find(requested_epoch);
            if (it != state_->insert_cache_per_epoch.end()) {
                const auto& cache = it->second;
                for (float v : cache.vectors) {
                    response->add_vectors(v);
                }
                for (int64_t id : cache.ids) {
                    response->add_ids(id);
                }
                dimension = cache.dimension;
                vector_count = cache.ids.size();
                has_cache = true;
            }
        }

        response->set_epoch(requested_epoch);
        response->set_vector_count(vector_count);
        response->set_dimension(dimension);
        response->set_has_cache(has_cache);

        std::cout << "{\"event\":\"GET_INSERT_CACHE\""
                  << ",\"timestamp_ms\":" << now_ms
                  << ",\"client_id\":\"" << client_id << "\""
                  << ",\"epoch\":" << requested_epoch
                  << ",\"vector_count\":" << vector_count
                  << ",\"has_cache\":" << (has_cache ? "true" : "false")
                  << "}" << std::endl;

        return Status::OK;
    }

    void set_state(ServerState* state) {
        state_ = state;
    }

private:
    void try_reclaim_old_epochs(uint64_t current_epoch) {
        // Reclaim epochs older than current where all clients have acked
        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        std::lock_guard<std::mutex> reclaim_lock(state_->reclaim_mutex);
        auto it = state_->pending_reclaim.begin();
        while (it != state_->pending_reclaim.end()) {
            uint64_t old_epoch = it->first;
            if (old_epoch < current_epoch &&
                state_->epoch_acks[old_epoch].size() >= registered_clients_->size()) {
                std::cout << "{\"event\":\"EPOCH_RECLAIMED\""
                          << ",\"timestamp_ms\":" << now_ms
                          << ",\"reclaimed_epoch\":" << old_epoch
                          << ",\"current_epoch\":" << current_epoch
                          << "}" << std::endl;
                it = state_->pending_reclaim.erase(it);
            } else {
                ++it;
            }
        }
    }


    std::vector<uint8_t>* serialized_meta_hnsw_;
    std::vector<size_t>* offset_sub_hnsw_;
    std::vector<size_t>* offset_para_;
    std::vector<size_t>* overflow_;
    std::vector<std::vector<dhnsw_idx_t>>* mapping_;
    ServerReconstructionManager* reconstruction_manager_;
    std::mutex* data_mutex_;

    // Reconstruction state pointers
    std::atomic<bool>* reconstruction_in_progress_;
    std::atomic<uint64_t>* current_reconstruction_id_;
    std::unordered_set<std::string>* acknowledged_clients_;
    std::mutex* ack_mutex_;
    std::condition_variable* ack_cv_;
    std::unordered_map<std::string, std::string>* registered_clients_;
    std::mutex* clients_mutex_;

    ReconstructionCoordinator* coordinator_ = nullptr;
    ServerState* state_ = nullptr;
};

// gRPC server runner
void RunGrpcServer(
    const std::string& server_address,
    ServerState* state,
    ReconstructionCoordinator* coordinator) {

    DhnswServiceWithReconstruction service(
        &state->serialized_meta_hnsw,
        &state->offset_sub_hnsw,
        &state->offset_para,
        &state->overflow,
        &state->mapping,
        state->reconstruction_manager.get(),
        &state->data_mutex,
        &state->reconstruction_in_progress,
        &state->current_reconstruction_id,
        &state->acknowledged_clients,
        &state->ack_mutex,
        &state->ack_cv,
        &state->registered_clients,
        &state->clients_mutex);

    // Set the coordinator for triggering reconstruction
    service.set_coordinator(coordinator);
    service.set_state(state);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.SetMaxReceiveMessageSize(INT_MAX);
    builder.SetMaxSendMessageSize(INT_MAX);
    // Increase server thread pool for better concurrency (handle 20+ worker threads)
    builder.SetSyncServerOption(ServerBuilder::SyncServerOption::NUM_CQS, 4);
    builder.SetSyncServerOption(ServerBuilder::SyncServerOption::MIN_POLLERS, 4);
    builder.SetSyncServerOption(ServerBuilder::SyncServerOption::MAX_POLLERS, 32);
    builder.RegisterService(&service);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "gRPC server listening on " << server_address << std::endl;

    server->Wait();
}

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
    bind_thread_to_cores(0, 0, 144);
    omp_set_num_threads(144);
    
    // Initialize server state
    ServerState state;
    
    // Build initial DistributedHnsw index
    int dim = FLAGS_dim;
    int num_meta = FLAGS_num_meta;
    int num_sub_hnsw = FLAGS_num_sub_hnsw;
    int meta_hnsw_neighbors = FLAGS_meta_hnsw_neighbors;
    int sub_hnsw_neighbors = FLAGS_sub_hnsw_neighbors;

    std::cout << "Loading dataset from: " << FLAGS_dataset_path << std::endl;
    
    int dim_, num_;
    auto base_data = read_fvecs(FLAGS_dataset_path, dim_, num_);
    std::cout << "Loaded " << num_ << " vectors of dimension " << dim_ << std::endl;

    DistributedHnsw dhnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, num_meta);
    dhnsw.build(base_data, 1000);
    
    std::cout << "Size of meta_hnsw: " << dhnsw.get_meta_hnsw_size() << " MB" << std::endl;
    
    // Serialize the index
    std::vector<size_t> offset_sub_hnsw;
    std::vector<size_t> offset_para;
    std::vector<size_t> overflow;
    std::vector<uint8_t> serialized_data = dhnsw.serialize_with_record_with_in_out_gap(
        offset_sub_hnsw, offset_para, overflow);
    
    std::cout << "Serialized data size: " << serialized_data.size() << " bytes" << std::endl;
    
    std::vector<uint8_t> serialized_meta_hnsw = dhnsw.serialize_meta_hnsw();
    std::vector<std::vector<dhnsw_idx_t>> mapping = dhnsw.get_mapping();
    
    // Store base vectors for reconstruction
    state.base_vectors = std::move(base_data);
    state.base_vector_count = num_;
    state.vector_dim = dim;

    // Store in server state
    state.serialized_meta_hnsw = serialized_meta_hnsw;
    state.offset_sub_hnsw = offset_sub_hnsw;
    state.offset_para = offset_para;
    state.overflow = overflow;
    state.mapping = mapping;
    state.serialized_data = serialized_data;
    
    // Initialize reconstruction manager
    state.reconstruction_manager = std::make_unique<ServerReconstructionManager>(
        dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, num_meta);
    state.reconstruction_manager->set_overflow_threshold(FLAGS_overflow_threshold);
    
    // Initialize RDMA resources
    RCtrl ctrl(FLAGS_rdma_port);
    RDMA_LOG(4) << "RDMA server listens at localhost:" << FLAGS_rdma_port;

    auto nic = RNic::create(RNicInfo::query_dev_names().at(FLAGS_use_nic_idx)).value();
    std::cout << "NIC created" << std::endl;
    
    RDMA_ASSERT(ctrl.opened_nics.reg(FLAGS_reg_nic_name, nic));

    // Allocate larger memory for double-buffering during reconstruction
    size_t total_size = serialized_data.size() * 3;  // 3x for double-buffering + headroom
    state.rdma_buffer_size = total_size;
    
    std::cout << "Total RDMA size: " << total_size << " bytes" << std::endl;
    RDMA_LOG(4) << "Allocating memory of size: " << total_size;

    state.mr_memory = Arc<RMem>(new RMem(total_size));
    RDMA_ASSERT(ctrl.registered_mrs.create_then_reg(FLAGS_reg_mem_name, state.mr_memory, nic));

    auto mr_attr = ctrl.registered_mrs.query(FLAGS_reg_mem_name).value()->get_reg_attr().value();
    state.rdma_buffer = reinterpret_cast<uint8_t*>(mr_attr.buf);
    
    // Copy initial serialized data into registered memory
    std::memcpy(state.rdma_buffer, serialized_data.data(), serialized_data.size());
    state.current_rdma_offset = 0;
    
    // Initialize epoch buffer manager for safe reconstruction
    state.epoch_buffer_manager = std::make_unique<EpochBufferManager>(
        state.rdma_buffer_size, state.rdma_buffer);
    
    // CRITICAL: Initialize epoch 0 buffer info - data starts at offset 0
    state.epoch_buffer_manager->init_epoch_zero(0, serialized_data.size());
    
    // Store initial epoch metadata
    {
        std::lock_guard<std::mutex> meta_lock(state.epoch_metadata_mutex);
        ServerState::EpochMetadata initial_meta;
        initial_meta.serialized_meta_hnsw = serialized_meta_hnsw;
        initial_meta.offset_sub_hnsw = offset_sub_hnsw;
        initial_meta.offset_para = offset_para;
        initial_meta.overflow = overflow;
        initial_meta.mapping = mapping;
        initial_meta.rdma_base_offset = 0;
        state.epoch_metadata[0] = std::move(initial_meta);
    }
    
    std::cout << "Data copied to RDMA buffer" << std::endl;
    std::cout << "Epoch buffer manager initialized with " << state.rdma_buffer_size << " bytes" << std::endl;
    
    // Initialize tiered insert buffer for reduced reconstruction frequency
    // The global gap is located after the initial serialized data
    state.tiered_insert_buffer = std::make_unique<TieredInsertBuffer>(
        FLAGS_num_sub_hnsw, FLAGS_dim, 0.1);
    
    // Set global gap: starts after initial data, ends at half buffer (other half for double-buffering)
    size_t global_gap_start = serialized_data.size();
    size_t global_gap_end = state.rdma_buffer_size / 2;  // First half is for epoch 0
    if (global_gap_end > global_gap_start) {
        state.tiered_insert_buffer->set_global_gap(global_gap_start, global_gap_end);
        std::cout << "Global gap initialized: " << global_gap_start << " - " << global_gap_end 
                  << " (" << (global_gap_end - global_gap_start) << " bytes)" << std::endl;
    }
    
    // Start RDMA daemon
    ctrl.start_daemon();
    std::cout << "MR key: " << mr_attr.key << std::endl;
    RDMA_LOG(4) << "RDMA resources initialized";

    // Start reconstruction coordinator
    ReconstructionCoordinator coordinator(&state, &ctrl);
    coordinator.start();
    
    // Start gRPC server
    std::string server_address = FLAGS_server_ip + ":" + std::to_string(FLAGS_port);
    std::thread grpc_server_thread(RunGrpcServer, server_address, &state, &coordinator);

    // Monitor reconstruction status periodically
    std::thread monitor_thread([&state]() {
        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(10));
            
            auto status = state.reconstruction_manager->get_status();
            if (status.phase != ReconstructionPhase::IDLE) {
                std::cout << "[Monitor] Reconstruction status: " 
                          << static_cast<int>(status.phase)
                          << " Progress: " << status.progress << "%" 
                          << " - " << status.message << std::endl;
            }
            
            // Check overflow levels
            bool needs_reconstruction = false;
            for (size_t i = 0; i < state.overflow.size() / 3; ++i) {
                double usage = calculate_overflow_usage(
                    std::vector<uint64_t>(state.overflow.begin(), state.overflow.end()), i);
                if (usage > FLAGS_overflow_threshold) {
                    std::cout << "[Monitor] Sub-HNSW " << i 
                              << " overflow usage: " << (usage * 100) << "%" << std::endl;
                    needs_reconstruction = true;
                }
            }
            
            if (needs_reconstruction && !state.reconstruction_in_progress.load()) {
                std::cout << "[Monitor] WARNING: Overflow threshold exceeded, "
                          << "reconstruction may be needed" << std::endl;
            }
        }
    });
    monitor_thread.detach();
    
    grpc_server_thread.join();

    return 0;
}

