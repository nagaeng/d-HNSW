#pragma once
#include <iostream>
#include <vector>
#include "../../deps/faiss/IndexFlat.h"
#include "../../deps/faiss/IndexHNSW.h"
#include "../../deps/faiss/MetaIndexes.h"
#include "../../deps/faiss/impl/HNSW.h"
#include "../../deps/faiss/Clustering.h"
#include <sstream>
#include <omp.h>
#include <faiss/utils/Heap.h>
#include <list>
#include "../../deps/rlib/core/lib.hh"
#include "../../deps/rlib/benchs/bench_op.hh"  
#include <gflags/gflags.h>
#include <cstdint> // for uint8_t
#include <grpcpp/grpcpp.h>
#include <grpcpp/grpcpp.h>
#include "../generated/dhnsw.grpc.pb.h"
#include "google/protobuf/empty.pb.h"
#include <dhnsw.pb.h>
#include "../../xcomm/src/rpc/mod.hh"
#include "../../xcomm/src/transport/rdma_ud_t.hh"
#include "../../deps/faiss/impl/io.h"
#include "../../xutils/huge_region.hh"
#include "dhnsw_io.h"
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <variant>
#include <functional>
#include <atomic> 
//TODO: consider restructuring the meta-hnsw
//TODO: update/insert/delete

// =============================================================================
// DEBUG FLAG: Disable runtime offset synchronization
// When set to 1, offsets (offset_subhnsw_, offset_para_, overflow_) are ONLY
// updated during reconstruction (via init()), NOT during normal search.
// This isolates whether segfaults are caused by epoch-based offset updates.
// Set to 0 to re-enable normal epoch-based offset synchronization.
// =============================================================================
#ifndef DISABLE_RUNTIME_OFFSET_SYNC
#define DISABLE_RUNTIME_OFFSET_SYNC 0  // Re-enabled: offset sync is required for concurrent insert+search
#endif

using dhnsw_idx_t = int64_t;
using namespace rdmaio;
using namespace rdmaio::rmem;
using namespace rdmaio::qp;
using namespace xstore::rpc;
using namespace r2;
using namespace xstore::util;
using namespace xstore::transport;

// prepare the sender transport
using SendTrait = UDTransport;
using RecvTrait = UDRecvTransport<2048>;
using SManager = UDSessionManager<2048>;

enum fetch_type {
        RDMA = 0,
        RDMA_DOORBELL = 1,
        RPC = 2,
        RPC_OVER_RDMA = 3,
    };

enum distance_type {
        Euclidean = 0,
        Angular = 1,
    };

struct CachedOffsets {
        std::vector<uint64_t> offset_sub_hnsw;
        std::vector<uint64_t> offset_para;
        std::vector<uint64_t> overflow;
        bool is_initialized;

        CachedOffsets() : is_initialized(false) {}
    };

struct AtomGroup {
    int meta_id;
    std::vector<dhnsw_idx_t> ids;      // global vector indices
};

struct Bin {
    std::vector<dhnsw_idx_t> ids;
    size_t size() const { return ids.size(); }
};


using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using dhnsw::DhnswService;
using dhnsw::Empty;
using dhnsw::MetaHnswResponse;
using dhnsw::Offset_SubHnswResponse;
using dhnsw::Offset_ParaResponse;
using dhnsw::MappingResponse;
using dhnsw::MappingEntry;
using dhnsw::OverflowResponse;

class DhnswClient {
public:

    DhnswClient(std::shared_ptr<Channel> channel)
        : stub_(DhnswService::NewStub(channel)) {}

    std::vector<uint8_t> GetMetaHnsw() {
        Empty request;
        MetaHnswResponse response;
        ClientContext context;

        Status status = stub_->GetMetaHnsw(&context, request, &response);
        if (status.ok()) {
            std::string data = response.serialized_meta_hnsw();
            return std::vector<uint8_t>(data.begin(), data.end());
        } else {
            std::cerr << "Failed to get meta_hnsw: " << status.error_message() << std::endl;
            return {};
        }
    }

    std::vector<size_t> GetOffset_sub_hnsw() {
        Empty request;
        dhnsw::Offset_SubHnswResponse response;
        ClientContext context;

        Status status = stub_->GetOffset_SubHnsw(&context, request, &response);
        if (status.ok()) {
            std::vector<size_t> offsets(response.offsets_subhnsw().begin(), response.offsets_subhnsw().end());
            return offsets;
        } else {
            std::cerr << "Failed to get offset: " << status.error_message() << std::endl;
            return {};
        }
    }

    std::vector<size_t> GetOffset_Para() {
        Empty request;
        dhnsw::Offset_ParaResponse response;
        ClientContext context;

        Status status = stub_->GetOffset_Para(&context, request, &response);
        if (status.ok()) {
            std::vector<size_t> offsets(response.offsets_para().begin(), response.offsets_para().end());
            return offsets;
        }
        else{
            std::cerr << "Failed to get offset: " << status.error_message() << std::endl;
            return {};
        }
    }

    std::vector<size_t> GetOverflow() {
        Empty request;
        dhnsw::OverflowResponse response;
        ClientContext context;

        Status status = stub_->GetOverflow(&context, request, &response);
        if (status.ok()) {
            std::vector<size_t> overflow(response.overflow().begin(), response.overflow().end());
            return overflow;
        }
        else{
            std::cerr << "Failed to get overflow: " << status.error_message() << std::endl;
            return {};
        }
    }


    std::vector<std::vector<dhnsw_idx_t>> GetMapping() {
        Empty request;
        MappingResponse response;
        ClientContext context;

        Status status = stub_->GetMapping(&context, request, &response);
        if (status.ok()) {
            std::vector<std::vector<dhnsw_idx_t>> mapping;
            for (const auto& entry : response.entries()) {
                std::vector<dhnsw_idx_t> sub_mapping(entry.mapping().begin(), entry.mapping().end());
                mapping.push_back(sub_mapping);
            }
            return mapping;
        } else {
            std::cerr << "Failed to get mapping: " << status.error_message() << std::endl;
            return {};
        }
    }

    // Reconstruction RPC methods
    struct ReconstructionResult {
        bool success;
        uint64_t reconstruction_id;
        std::string message;
        int phase;  // ReconstructionPhase enum value
        double progress;
    };

    ReconstructionResult TriggerReconstruction(
            const std::vector<float>& buffered_vectors,
            int vector_count,
            const std::string& client_id) {
        dhnsw::TriggerReconstructionRequest request;
        request.set_client_id(client_id);
        request.set_vector_count(vector_count);
        // Serialize buffered vectors
        for (float v : buffered_vectors) {
            request.add_buffered_vectors(v);
        }

        dhnsw::TriggerReconstructionResponse response;
        ClientContext context;

        // Set timeout for reconstruction (may take a while)
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(300));

        Status status = stub_->TriggerReconstruction(&context, request, &response);

        ReconstructionResult result;
        if (status.ok()) {
            result.success = response.success();
            result.reconstruction_id = response.reconstruction_id();
            result.message = response.message();
            std::cout << "[RPC] TriggerReconstruction response: success=" << result.success
                      << " id=" << result.reconstruction_id
                      << " msg=" << result.message << std::endl;
        } else {
            result.success = false;
            result.reconstruction_id = 0;
            result.message = "RPC failed: " + status.error_message();
            std::cerr << "[RPC] TriggerReconstruction failed: " << status.error_message() << std::endl;
        }
        return result;
    }

    ReconstructionResult GetReconstructionStatus() {
        Empty request;
        dhnsw::ReconstructionStatusResponse response;
        ClientContext context;

        Status status = stub_->GetReconstructionStatus(&context, request, &response);

        ReconstructionResult result;
        if (status.ok()) {
            result.success = true;
            result.reconstruction_id = response.reconstruction_id();
            result.phase = response.phase();
            result.progress = response.progress();
            result.message = response.message();
        } else {
            result.success = false;
            result.message = "RPC failed: " + status.error_message();
        }
        return result;
    }

    bool AcknowledgeReconstruction(const std::string& client_id, uint64_t reconstruction_id, uint64_t epoch = 0) {
        dhnsw::AckReconstructionRequest request;
        request.set_client_id(client_id);
        request.set_reconstruction_id(reconstruction_id);
        request.set_epoch(epoch);

        dhnsw::AckReconstructionResponse response;
        ClientContext context;

        Status status = stub_->AcknowledgeReconstruction(&context, request, &response);

        auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        std::cout << "{\"event\":\"ACK_EPOCH\""
                  << ",\"timestamp_ms\":" << now_ms
                  << ",\"client_id\":\"" << client_id << "\""
                  << ",\"epoch\":" << epoch
                  << ",\"reconstruction_id\":" << reconstruction_id
                  << ",\"all_acknowledged\":" << (response.all_acknowledged() ? "true" : "false")
                  << "}" << std::endl;

        if (status.ok()) {
            return response.all_acknowledged();
        } else {
            std::cerr << "[RPC] AcknowledgeReconstruction failed: " << status.error_message() << std::endl;
            return false;
        }
    }

    // Epoch info for RDMA buffer coordination
    struct EpochInfo {
        bool success;  // Whether the RPC succeeded (NOT based on returned values)
        uint64_t epoch;
        uint64_t rdma_offset;
        uint64_t reconstruction_id;
        bool reconstruction_in_progress;
    };

    EpochInfo GetEpochInfo() {
        Empty request;
        dhnsw::EpochInfoResponse response;
        ClientContext context;

        Status status = stub_->GetEpochInfo(&context, request, &response);

        EpochInfo info{false, 0, 0, 0, false};
        if (status.ok()) {
            info.success = true;  // RPC succeeded - epoch=0, offset=0 is valid initial state!
            info.epoch = response.epoch();
            info.rdma_offset = response.rdma_offset();
            info.reconstruction_id = response.reconstruction_id();
            info.reconstruction_in_progress = response.reconstruction_in_progress();
        } else {
            info.success = false;  // RPC failed
            std::cerr << "[RPC] GetEpochInfo failed: " << status.error_message() << std::endl;
        }
        return info;
    }

    // Get insert cache from master during reconstruction
    struct InsertCacheData {
        uint64_t epoch;
        std::vector<float> vectors;
        std::vector<int64_t> ids;
        int vector_count;
        int dimension;
        bool has_cache;
    };

    InsertCacheData GetInsertCache(const std::string& client_id, uint64_t epoch) {
        dhnsw::GetInsertCacheRequest request;
        request.set_client_id(client_id);
        request.set_epoch(epoch);

        dhnsw::InsertCacheResponse response;
        ClientContext context;

        Status status = stub_->GetInsertCache(&context, request, &response);

        InsertCacheData data{0, {}, {}, 0, 0, false};
        if (status.ok()) {
            data.epoch = response.epoch();
            data.vectors.assign(response.vectors().begin(), response.vectors().end());
            data.ids.assign(response.ids().begin(), response.ids().end());
            data.vector_count = response.vector_count();
            data.dimension = response.dimension();
            data.has_cache = response.has_cache();

            auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            std::cout << "{\"event\":\"GET_INSERT_CACHE\""
                      << ",\"timestamp_ms\":" << now_ms
                      << ",\"client_id\":\"" << client_id << "\""
                      << ",\"epoch\":" << epoch
                      << ",\"vector_count\":" << data.vector_count
                      << ",\"has_cache\":" << (data.has_cache ? "true" : "false")
                      << "}" << std::endl;
        } else {
            std::cerr << "[RPC] GetInsertCache failed: " << status.error_message() << std::endl;
        }
        return data;
    }

    // Epoch-based RDMA read coordination
    // Acquire a read reference before doing RDMA reads
    // Returns ALL metadata for the epoch atomically - no separate RPCs needed
    struct EpochReadInfo {
        bool success;
        uint64_t epoch;
        uint64_t rdma_base_offset;
        bool has_metadata;
        std::vector<uint64_t> offset_subhnsw;
        std::vector<uint64_t> offset_para;
        std::vector<uint64_t> overflow;
        std::vector<uint8_t> serialized_meta_hnsw;
        std::vector<std::vector<dhnsw_idx_t>> mapping;
    };

    EpochReadInfo AcquireEpochRead(const std::string& client_id) {
        dhnsw::AcquireEpochRequest request;
        request.set_client_id(client_id);

        dhnsw::AcquireEpochResponse response;
        ClientContext context;
        
        // Set a 5-second deadline to prevent hanging indefinitely
        context.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(5));

        Status status = stub_->AcquireEpochRead(&context, request, &response);

        EpochReadInfo info{false, 0, 0, false, {}, {}, {}, {}, {}};
        if (status.ok()) {
            info.success = response.success();
            info.epoch = response.epoch();
            info.rdma_base_offset = response.rdma_base_offset();
            info.has_metadata = response.has_metadata();
            if (info.has_metadata) {
                info.offset_subhnsw.assign(response.offset_subhnsw().begin(),
                                           response.offset_subhnsw().end());
                info.offset_para.assign(response.offset_para().begin(),
                                        response.offset_para().end());
                info.overflow.assign(response.overflow().begin(),
                                     response.overflow().end());
                
                // Serialized meta HNSW
                const std::string& meta_str = response.serialized_meta_hnsw();
                info.serialized_meta_hnsw.assign(meta_str.begin(), meta_str.end());
                
                // Mapping - entries are already in sub_index order
                for (const auto& entry : response.mapping()) {
                    std::vector<dhnsw_idx_t> m;
                    for (int64_t idx : entry.mapping()) {
                        m.push_back(static_cast<dhnsw_idx_t>(idx));
                    }
                    info.mapping.push_back(std::move(m));
                }
            }
        } else {
            std::cerr << "[RPC] AcquireEpochRead failed: " << status.error_message() << std::endl;
        }
        return info;
    }

    // Release a read reference after RDMA reads complete
    bool ReleaseEpochRead(const std::string& client_id, uint64_t epoch) {
        dhnsw::ReleaseEpochRequest request;
        request.set_client_id(client_id);
        request.set_epoch(epoch);

        dhnsw::ReleaseEpochResponse response;
        ClientContext context;

        Status status = stub_->ReleaseEpochRead(&context, request, &response);

        if (status.ok()) {
            return response.success();
        } else {
            std::cerr << "[RPC] ReleaseEpochRead failed: " << status.error_message() << std::endl;
            return false;
        }
    }

private:
    std::unique_ptr<DhnswService::Stub> stub_;
};


class DhnswServiceImpl final : public DhnswService::Service {
public:
    DhnswServiceImpl(const std::vector<uint8_t>& serialized_meta_hnsw,
                     const std::vector<size_t>& offset_sub_hnsw,
                     const std::vector<size_t>& offset_para,
                     const std::vector<size_t>& overflow,
                     const std::vector<std::vector<dhnsw_idx_t>>& mapping)
        : serialized_meta_hnsw_(serialized_meta_hnsw), offset_sub_hnsw_(offset_sub_hnsw), offset_para_(offset_para), overflow_(overflow), mapping_(mapping) {}

    Status GetMetaHnsw(ServerContext* context, const Empty* request,
                       MetaHnswResponse* response) override {
        response->set_serialized_meta_hnsw(
            std::string(serialized_meta_hnsw_.begin(), serialized_meta_hnsw_.end()));
        return Status::OK;
    }

    Status GetOffset_SubHnsw(ServerContext* context, const Empty* request,
                     Offset_SubHnswResponse* response) override {
        for (size_t off : offset_sub_hnsw_) {
            response->add_offsets_subhnsw(off);
        }
        return Status::OK;
    }

    Status GetOffset_Para(ServerContext* context, const Empty* request,
                     Offset_ParaResponse* response) override {
        for (size_t off : offset_para_) {
            response->add_offsets_para(off);
        }
        return Status::OK;
    }

    Status GetMapping(ServerContext* context, const Empty* request,
                      MappingResponse* response) override {
        for (size_t i = 0; i < mapping_.size(); ++i) {
            MappingEntry* entry = response->add_entries();
            entry->set_sub_index(static_cast<uint32_t>(i));
            for (dhnsw_idx_t idx : mapping_[i]) {
                entry->add_mapping(idx);
            }
        }
        return Status::OK;
    }

    Status GetOverflow(ServerContext* context, const Empty* request,
                     OverflowResponse* response) override {
        for (size_t off : overflow_) {
            response->add_overflow(off);
        }
        return Status::OK;
    }

    // Epoch-based RDMA read coordination for static (non-reconstruction) server.
    // Always returns epoch 0 with rdma_base_offset 0 and all metadata inline.
    Status AcquireEpochRead(ServerContext* context,
                            const dhnsw::AcquireEpochRequest* request,
                            dhnsw::AcquireEpochResponse* response) override {
        response->set_success(true);
        response->set_epoch(0);
        response->set_rdma_base_offset(0);
        response->set_has_metadata(true);

        // Inline offset_subhnsw
        for (size_t off : offset_sub_hnsw_) {
            response->add_offset_subhnsw(off);
        }
        // Inline offset_para
        for (size_t off : offset_para_) {
            response->add_offset_para(off);
        }
        // Inline overflow
        for (size_t off : overflow_) {
            response->add_overflow(off);
        }
        // Inline serialized meta_hnsw
        response->set_serialized_meta_hnsw(
            std::string(serialized_meta_hnsw_.begin(), serialized_meta_hnsw_.end()));
        // Inline mapping
        for (size_t i = 0; i < mapping_.size(); ++i) {
            dhnsw::MappingEntry* entry = response->add_mapping();
            entry->set_sub_index(static_cast<uint32_t>(i));
            for (dhnsw_idx_t idx : mapping_[i]) {
                entry->add_mapping(idx);
            }
        }
        return Status::OK;
    }

    Status ReleaseEpochRead(ServerContext* context,
                            const dhnsw::ReleaseEpochRequest* request,
                            dhnsw::ReleaseEpochResponse* response) override {
        // Static server: no epoch tracking needed, always succeed
        response->set_success(true);
        return Status::OK;
    }

    // Status GetSubHnsw(ServerContext* context, const SubHnswRequest* request,
    //                  SubHnswResponse* response) override {
    //     uint32_t sub_index = request->sub_index();

    //     // Calculate start and end positions using offset_
    //     uint64_t start_pos = offset_[sub_index * 2];      // Begin position
    //     uint64_t end_pos = offset_[sub_index * 2 + 1];    // End position

    //     // Validate positions
    //     if (start_pos > end_pos || end_pos > subhnsw_data_.size()) {
    //         return Status(grpc::StatusCode::INVALID_ARGUMENT, "Invalid offset positions");
    //     }

    //     size_t length = end_pos - start_pos;

    //     // Extract the serialized data
    //     const uint8_t* data_ptr = subhnsw_data_.data() + start_pos;
    //     std::string serialized_sub_hnsw(reinterpret_cast<const char*>(data_ptr), length);

    //     // Set the response
    //     response->set_serialized_sub_hnsw(serialized_sub_hnsw);

    //     return Status::OK; 
    // }

private:
    std::vector<uint8_t> serialized_meta_hnsw_;
    std::vector<size_t> offset_sub_hnsw_;
    std::vector<size_t> offset_para_;
    std::vector<size_t> overflow_;
    std::vector<std::vector<dhnsw_idx_t>> mapping_;
    std::vector<int> part_;
    // std::vector<uint8_t> subhnsw_data_;
};


class DistributedHnsw {
    private:
        int d;  // Dimensionality of the vectors
        int num_sub_hnsw;  // Number of sub-indices
        int meta_M;  // HNSW parameter: number of neighbors to store in each layer
        int sub_M;  // HNSW parameter: number of neighbors to store in each layer
        faiss::IndexHNSWFlat* meta_hnsw;  // Meta-HNSW index pointer
        
        int num_meta;  // Number of vectors in the meta-hnsw
        std::vector<std::vector<dhnsw_idx_t>> mapping; // mapping from sub-hnsw index to original index
        std::vector<uint64_t> record;

        // Dynamic Overflow Management
        size_t overflow_capacity; // Tracks current number of overflow entries
        std::mutex overflow_mutex; // Mutex for thread-safe operations
    public: 
        std::vector<faiss::IndexHNSWFlat*> sub_hnsw;  // Vector to hold sub-indices   
        // initialization
        DistributedHnsw();

        DistributedHnsw(int dim, int sub_partitions, int meta_hnsw_neighbors, int sub_hnsw_neighbors, int num_meta, distance_type flag = Euclidean) 
            : d(dim), num_sub_hnsw(sub_partitions), meta_M(meta_hnsw_neighbors), sub_M(sub_hnsw_neighbors),num_meta(num_meta){
            if(flag == Euclidean){
            meta_hnsw = new faiss::IndexHNSWFlat(d, meta_M);  // Initialize meta-index
            }
            else if(flag == Angular){
            meta_hnsw = new faiss::IndexHNSWFlat(d, meta_M, faiss::METRIC_INNER_PRODUCT);  // Initialize meta-index
            }
            else{
                std::cerr << "Invalid distance type" << std::endl; 
            }
            sub_hnsw.resize(num_sub_hnsw, nullptr); // Initialize sub-indices
            if(flag == 0) {
                for (int i = 0; i < num_sub_hnsw; i++) {
                    sub_hnsw[i] = new faiss::IndexHNSWFlat(d, sub_M);  // Initialize each sub-index
                } 
            }
            else{
                for (int i = 0; i < num_sub_hnsw; i++) {
                    sub_hnsw[i] = new faiss::IndexHNSWFlat(d, sub_M, faiss::METRIC_INNER_PRODUCT);  // Initialize each sub-index
                } 
            }
            mapping.resize(num_sub_hnsw);
        }

        // Destructor to clean up memory
        ~DistributedHnsw() {
            if (meta_hnsw) {
                delete meta_hnsw;  // Delete meta-index
                meta_hnsw = nullptr; 
            }
            for (auto& index : sub_hnsw) {
            if (index) {
                delete index;  // Delete each sub-index
                index = nullptr; 
            }
            }
            sub_hnsw.clear();
        }

        DistributedHnsw(const DistributedHnsw& other);
        // Method to build
        void build(
            const std::vector<float>& data,
            size_t target
        ); //data is flat

        void print_subhnsw_balance() const;
        
        size_t get_meta_hnsw_size() const;

        size_t get_per_sub_hnsw_size() const;

        // Method to search meta-hnsw,result record in sub_hnsw_tosearch
        void meta_search(
            const int n,
            const float* query, 
            int K_meta, 
            float* distances, 
            dhnsw_idx_t* labels, 
            std::vector<int>& sub_hnsw_tosearch
        );

        // Method to search sub-hnsw, result record in distances and labels (Global view / single machine)
        void sub_search(
            const int n,
            const float* query,
            int K_meta, 
            int K_sub, 
            float* distances, 
            dhnsw_idx_t* labels, 
            std::vector<int>& sub_hnsw_tosearch,
            dhnsw_idx_t* tmp_sub_hnsw_tags
        );

        // Method to search using the hierarchical index, result record in distances and labels
        void hierarchicalSearch(
            const int n,
            const float* query, 
            int K_meta, 
            int K_sub, 
            float* distances, 
            dhnsw_idx_t* labels,
            dhnsw_idx_t* sub_hnsw_tags,
            dhnsw_idx_t* original_index,
            int ef
        );
        
        void insert(
            const int n, 
            const std::vector<float>& data
        );

        std::vector<uint8_t> serialize_with_record_with_in_out_gap(
            std::vector<uint64_t>& offset_sub_hnsw, 
            std::vector<uint64_t>& offset_para, 
            std::vector<uint64_t>& overflow
        ) const ;

        DistributedHnsw deserialize_with_record_with_in_out_gap(
            const std::vector<uint8_t>& data,
            std::vector<uint64_t>& offset_sub_hnsw, 
            std::vector<uint64_t>& offset_para, 
            std::vector<uint64_t>& overflow
        ) const ;

        faiss::IndexHNSWFlat* deserialize_sub_hnsw_with_record_with_in_out_gap(
            const std::vector<uint8_t>& data,
            int sub_idx,
            std::vector<uint64_t>& offset_sub_hnsw,
            std::vector<uint64_t>& offset_para, 
            std::vector<uint64_t>& overflow
        ) const;

        void serialize_with_record_with_gap_to_file(
            const std::string& filename,
            std::vector<uint64_t>& offset  
        ); 
        // Method to serialize the whole index
        std::vector<uint8_t> serialize_with_record( 
            std::vector<uint64_t>& offset
        ) const ;

        std::vector<uint8_t> serialize_with_record_with_gap( 
            std::vector<uint64_t>& offset
        ) const ;
        
       std::vector<uint8_t> initial_serialize_whole( 
            std::vector<uint64_t>& offset_sub_hnsw,
            std::vector<uint64_t>& offset_para,
            std::vector<uint64_t>& overflow
        ) const ;

        static DistributedHnsw initial_deserialize_whole(
            const std::vector<uint8_t>& data,
            const std::vector<uint64_t>& offset_sub_hnsw,
            const std::vector<uint64_t>& offset_para,
            const std::vector<uint64_t>& overflow
        );

        void append_xb_data(
            faiss::IndexHNSWFlat* index, 
            const std::vector<uint8_t>& overflow_data, 
            size_t d
        );

        void append_neighbors_data(
            faiss::IndexHNSWFlat* index, 
            const std::vector<uint8_t>& overflow_data
        );

        void append_offsets_data(
            faiss::IndexHNSWFlat* index, 
            const std::vector<uint8_t>& overflow_data
        );

        void append_levels_data(
            faiss::IndexHNSWFlat* index, 
            const std::vector<uint8_t>& overflow_data
        );
        // Method to serialize the whole index
        std::vector<uint8_t> serialize_insert_with_record( 
            std::vector<uint64_t>& offset
        ) const ;

        std::vector<uint8_t> serialize_insert_with_record_with_gap( 
            std::vector<uint64_t>& offset
        ) const ;

        void write_dhnsw (const DistributedHnsw *idx, faiss::IOWriter *f);

        void read_dhnsw (DistributedHnsw *idx, faiss::IOReader *f);

        void insert_with_record(const int n, 
                    const std::vector<float>& data);

        std::vector<uint8_t> serialize() const ;

        // Method to deserialize the whole index
        static DistributedHnsw deserialize(
            const std::vector<uint8_t>& data
        );

        // Method to serialize the meta_hnsw
        std::vector<uint8_t> serialize_meta_hnsw() const ; 

        // Method to deserialize the meta_hnsw
        static faiss::IndexHNSWFlat* deserialize_meta_hnsw( 
            const std::vector<uint8_t>& data
        );

        // Method to serialize the sub_hnsw
        std::vector<uint8_t> serialize_sub_hnsw(
            int sub_index //which sub-index
        ) const ; 

        // Method to deserialize the sub_hnsw
        static faiss::IndexHNSWFlat* deserialize_sub_hnsw( 
            const std::vector<uint8_t>& data
        );

        // Method to deserialize the sub_hnsw with serialized_dhnsw_data and offset and sub_hnsw_tosearch
        static std::vector<faiss::IndexHNSWFlat*> deserialize_sub_hnsw_batch( 
            const std::vector<uint8_t>& data, 
            const std::vector<uint64_t>& offset,
            std::vector<int> sub_hnsw_tosearch
        );

        // Method to serialize vector
        std::vector<uint8_t> serialize_offset(
            const std::vector<uint64_t>& data
        ) const ;

        // Method to deserialize vector
        static std::vector<uint64_t> deserialize_offset(
            const std::vector<uint8_t>& data
        );
        std::vector<uint8_t> serialize_part() const ;

        static std::vector<int> deserialize_part(
            const std::vector<uint8_t>& data
        );

        std::vector<uint8_t> serialize_initialsend(std::vector<uint64_t>& offset) const ;
        //Method to serialize for client

        std::vector<uint8_t> serialize4client() const ; ////for 2-side rdma

        void deserialize4client(
            const std::vector<uint8_t>& data,
            int& dim, 
            int& num_sub_hnsw, 
            int& meta_M, 
            int& sub_M, 
            int& num_meta,
            faiss::IndexHNSWFlat* meta_hnsw 
        );
        faiss::IndexHNSWFlat* get_meta_hnsw();

        std::vector<faiss::IndexHNSWFlat*> get_sub_hnsw(std::vector<int> local_sub_hnsw); 
        
        
        std::vector<std::vector<dhnsw_idx_t>> get_mapping();
        
        
        DistributedHnsw deserialize_with_record_with_gap(
            const std::vector<uint8_t>& data,
            const std::vector<uint64_t>& offset
        );

        
    
};

struct FetchHnswRPCRequest {
    std::vector<int> sub_indices;
    
    void serialize(char* buf, size_t& offset) const {
        size_t vec_size = sub_indices.size();
        memcpy(buf + offset, &vec_size, sizeof(size_t));
        offset += sizeof(size_t);
        memcpy(buf + offset, sub_indices.data(), vec_size * sizeof(int));
        offset += vec_size * sizeof(int);
    }
    
    static FetchHnswRPCRequest deserialize(const char* buf, size_t& offset) {
        FetchHnswRPCRequest req;
        size_t vec_size;
        memcpy(&vec_size, buf + offset, sizeof(size_t));
        offset += sizeof(size_t);
        req.sub_indices.resize(vec_size);
        memcpy(req.sub_indices.data(), buf + offset, vec_size * sizeof(int));
        offset += vec_size * sizeof(int);
        return req;
    }
};

struct FetchHnswRPCResponse {
    std::vector<std::pair<uint64_t, uint64_t>> index_locations;  // (offset, size) pairs
    
    void serialize(char* buf, size_t& offset) const {
        size_t vec_size = index_locations.size();
        memcpy(buf + offset, &vec_size, sizeof(size_t));
        offset += sizeof(size_t);
        memcpy(buf + offset, index_locations.data(), 
               vec_size * sizeof(std::pair<uint64_t, uint64_t>));
        offset += vec_size * sizeof(std::pair<uint64_t, uint64_t>);
    }
    
    static FetchHnswRPCResponse deserialize(const char* buf, size_t& offset) {
        FetchHnswRPCResponse resp;
        size_t vec_size;
        memcpy(&vec_size, buf + offset, sizeof(size_t));
        offset += sizeof(size_t);
        resp.index_locations.resize(vec_size);
        memcpy(resp.index_locations.data(), buf + offset,
               vec_size * sizeof(std::pair<uint64_t, uint64_t>));
        offset += vec_size * sizeof(std::pair<uint64_t, uint64_t>);
        return resp;
    }
};

class SimpleAllocator : public rdmaio::qp::AbsRecvAllocator {
  RMem::raw_ptr_t buf = nullptr;
  usize total_mem = 0;
  mr_key_t key;

  RegAttr mr;

public:
  SimpleAllocator(Arc<RMem> mem, const RegAttr &mr)
      : buf(mem->raw_ptr), total_mem(mem->sz), mr(mr), key(mr.key) {
    // RDMA_LOG(4) << "simple allocator use key: " << key;
  }

  ::r2::Option<std::pair<rmem::RMem::raw_ptr_t, rmem::mr_key_t>>
  alloc_one(const usize &sz) override {
    if (total_mem < sz)
      return {};
    auto ret = buf;
    buf = static_cast<char *>(buf) + sz;
    total_mem -= sz;
    return std::make_pair(ret, key);
  }

  ::rdmaio::Option<std::pair<rmem::RMem::raw_ptr_t, rmem::RegAttr>>
  alloc_one_for_remote(const usize &sz) override {
    if (total_mem < sz)
      return {};
    auto ret = buf;
    buf = static_cast<char *>(buf) + sz;
    total_mem -= sz;
    return std::make_pair(ret, mr);
  }
};

// Thread-safe queue for communication between fetch and search threads
template <typename T>
class ThreadSafeQueue {
public:
    void push(const T& item) {
        std::lock_guard<std::mutex> lg(mutex_);
        queue_.push(item);
        cv_.notify_one();
    }
    T pop() {
        std::unique_lock<std::mutex> ul(mutex_);
        cv_.wait(ul, [&]{ return !queue_.empty() || stopped_; });
        if (stopped_ && queue_.empty()) {
            // std::cout << "queue is empty, return -1" << std::endl;
            if constexpr (std::is_same_v<T, int>) {
                return static_cast<T>(-1);
            } else {
                return std::make_pair(-1, nullptr);
            }
        }
        T item = queue_.front(); queue_.pop();
        return item;
    }
    void stop() {
        std::lock_guard<std::mutex> lg(mutex_);
        stopped_ = true;
        cv_.notify_all();
        queue_ = std::queue<T>();
    }
    bool empty() {
        std::lock_guard<std::mutex> lg(mutex_);
        return queue_.empty();
    }
    void suspend() {
        std::lock_guard<std::mutex> lg(mutex_);
        stopped_ = true;
        cv_.notify_all();
        queue_ = std::queue<T>();
    }
    void restart() {
        std::lock_guard<std::mutex> lg(mutex_);
        stopped_ = false;
        cv_.notify_all();
    }
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopped_ = false;
};

struct CacheEntry {
    std::vector<uint8_t> raw_bytes;

    std::shared_ptr<faiss::IndexHNSWFlat> idx_ptr;

    CacheEntry(std::vector<uint8_t>&& bytes)
      : raw_bytes(std::move(bytes)),
        idx_ptr(nullptr)
    {}
};

// LRU Cache for sub-HNSW shards
class LRUCache {
    friend class LocalHnsw;
    friend class DistributedHnsw;
public:
    LRUCache(size_t capacity) : capacity_(capacity) {}
    // faiss::IndexHNSWFlat* get(int key) {
    //     std::lock_guard<std::mutex> lg(m_);
    //     auto it = cache_map_.find(key);
    //     if (it == cache_map_.end()) return nullptr;
    //     // move to front
    //     lru_list_.splice(lru_list_.begin(), lru_list_, it->second.second);
    //     used_indices_.insert(key);
    //     return it->second.first;
    // }
    faiss::IndexHNSWFlat* get(int key) {
        std::lock_guard<std::mutex> lg(m_);
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) return nullptr;
        // move to front
        lru_list_.splice(lru_list_.begin(), lru_list_, std::get<1>(it->second));
        used_indices_.insert(key);
        return std::get<2>(it->second);
    }
    std::pair<std::vector<uint8_t>, faiss::IndexHNSWFlat*> get_serialized_and_index(int key) {
        std::lock_guard<std::mutex> lg(m_);
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) return std::make_pair(std::vector<uint8_t>(), nullptr);
        return std::make_pair(std::get<0>(it->second), std::get<2>(it->second)  );
    }
    bool has_serialized_or_index(int key) {
        std::lock_guard<std::mutex> lg(m_);
        return cache_map_.count(key) > 0;
    }
    // int check_and_evict() { //return the remaining capacity, evict failed if return 0, evict all can evict|| ? no inter batch reuse
    bool check_and_evict() {
        std::lock_guard<std::mutex> lg(m_);

        if (cache_map_.size() < capacity_) {
            // return capacity_ - cache_map_.size(); 
            return true;
        }
        int old = -1;
        for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
            if (in_use_indices_.count(*it) == 0 && used_indices_.count(*it) > 0) {
                old = *it;
                break;
            }
        }
        if (old < 0) {
            // return 0;
            return false;
        }

        for (auto it = lru_list_.begin(); it != lru_list_.end(); ++it) {
            if (*it == old) {
                lru_list_.erase(it);
                break;
            }
        }
        auto ev = cache_map_.find(old);
        // delete ev->second.first;
        delete std::get<2>(ev->second);
        std::get<0>(ev->second).clear();
        cache_map_.erase(ev);
        return true;
    }

    // bool put(int key, faiss::IndexHNSWFlat* val) {
    //     std::lock_guard<std::mutex> lg(m_);
    //     // if exist, update
    //     auto it = cache_map_.find(key);
    //     if (it != cache_map_.end()) {
    //         // delete it->second.first;
    //         it->second.first.clear();
    //         lru_list_.erase(it->second.second);
    //         cache_map_.erase(it);
    //     }
        
    //     // evict if needed 
    //     if (cache_map_.size() >= capacity_) {
    //         int old = -1;
    //         if(used_indices_.empty()) {
    //             old = lru_list_.back();
    //             lru_list_.pop_back();
    //             auto ev = cache_map_.find(old);
    //             if (ev != cache_map_.end() && in_use_indices_.count(old) == 0) {
    //                 // delete ev->second.first;
    //                 ev->second.first.clear();
    //                 cache_map_.erase(ev);
    //             }
    //         }
    //         else{
    //             // find the least recently used sub-HNSW which has been used
    //             for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
    //                 if (in_use_indices_.count(*it)  == 0 && used_indices_.count(*it) > 0) {
    //                     old = *it;
    //                     lru_list_.erase(std::next(it).base());
    //                     break;
    //                 }
    //             }
    //             if (old != -1) {
    //                 auto ev = cache_map_.find(old);
    //                 if (ev != cache_map_.end()) {
    //                     // delete ev->second.first;
    //                     ev->second.first.clear();
    //                     cache_map_.erase(ev);
    //                 }
    //             } else {
    //                 std::cerr << "cache is full, but evict failed" << std::endl;
    //                 return false;
    //             }
    //         }
    //     }
        
    //     lru_list_.push_front(key);
    //     cache_map_[key] = {val, lru_list_.begin()}; 
    //     return true;
    // }
    bool put(int key, std::vector<uint8_t> val) {
        std::lock_guard<std::mutex> lg(m_);
        // if exist, update
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            // delete it->second.first;
            std::get<0>(it->second).clear();
            delete std::get<2>(it->second);
            lru_list_.erase(std::get<1>(it->second));
            cache_map_.erase(it);
        }
        
        // evict if needed 
        if (cache_map_.size() >= capacity_) {
            int old = -1;
            if(used_indices_.empty()) {
                old = lru_list_.back();
                lru_list_.pop_back();
                auto ev = cache_map_.find(old);
                if (ev != cache_map_.end() && in_use_indices_.count(old) == 0 && used_indices_.count(old) > 0) {
                    // delete ev->second.first;
                    std::get<0>(ev->second).clear();
                    cache_map_.erase(ev);
                }
            }
            else{
                // find the least recently used sub-HNSW which has been used
                for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
                    if (in_use_indices_.count(*it)  == 0 && used_indices_.count(*it) > 0) {
                        old = *it;
                        lru_list_.erase(std::next(it).base());
                        break;
                    }
                }
                if (old != -1) {
                    auto ev = cache_map_.find(old);
                    if (ev != cache_map_.end()) {
                        // delete ev->second.first;
                        delete std::get<2>(ev->second);
                        std::get<0>(ev->second).clear();
                        cache_map_.erase(ev);
                    }
                } else {
                    std::cerr << "cache is full, but evict failed" << std::endl;
                    return false;
                }
            }
        }
        
        lru_list_.push_front(key);
        cache_map_[key] = {val, lru_list_.begin(), nullptr}; 
        return true;
    }
    void update(int key, faiss::IndexHNSWFlat* val) {
        std::lock_guard<std::mutex> lg(m_);
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            std::get<2>(it->second) = val;
        }
    }

    void idle_update() {
    std::vector<int> keys_to_deserialize;
    std::vector<std::vector<uint8_t>> serialized_data_copy;

    {
        std::lock_guard<std::mutex> lg(m_);
        for (auto const& kv : cache_map_) { 
            if (std::get<2>(kv.second) == nullptr &&
                in_use_indices_.count(kv.first) == 0 &&
                std::get<0>(kv.second).size() > 0) { 
                keys_to_deserialize.push_back(kv.first);
                serialized_data_copy.push_back(std::get<0>(kv.second)); 
                break;
            }
        }
    } 

    if (!keys_to_deserialize.empty() && !serialized_data_copy.empty()) {
        for(int i = 0; i < keys_to_deserialize.size(); i++) {
            faiss::IndexHNSWFlat* deserialized_index = DistributedHnsw::deserialize_sub_hnsw(serialized_data_copy[i]);

        if (deserialized_index) { 
            std::lock_guard<std::mutex> lg(m_);
            auto it = cache_map_.find(keys_to_deserialize[i]);
            if (it != cache_map_.end() && std::get<2>(it->second) == nullptr) {
                std::get<2>(it->second) = deserialized_index;
            } else if (it == cache_map_.end()) {
                delete deserialized_index;
            }
            else if (std::get<2>(it->second) != nullptr && std::get<2>(it->second) != deserialized_index) {
                 delete deserialized_index;
            }
        }
    }
    } 
    }
    bool has_serialized_and_index(int key) {
        std::lock_guard<std::mutex> lg(m_);
        return cache_map_.count(key) > 0 && std::get<0>(cache_map_[key]).size() > 0 && std::get<2>(cache_map_[key]) != nullptr;
    }
    bool has_serialized_not_index(int key) {
        std::lock_guard<std::mutex> lg(m_);
        return cache_map_.count(key) > 0 && std::get<0>(cache_map_[key]).size() > 0 && std::get<2>(cache_map_[key]) == nullptr;
    }
    void clear() {
        std::lock_guard<std::mutex> lg(m_);
        // for (auto &kv : cache_map_) delete kv.second.first;
        for (auto &kv : cache_map_) {
            delete std::get<2>(kv.second);
            std::get<0>(kv.second).clear();
        }
        cache_map_.clear();
        lru_list_.clear();
        in_use_indices_.clear();
        used_indices_.clear();
    }
    
private:
    size_t capacity_;
    std::list<int> lru_list_;
    // std::unordered_map<int, std::pair<faiss::IndexHNSWFlat*, std::list<int>::iterator>> cache_map_;
    std::unordered_map<int, std::tuple<std::vector<uint8_t>, std::list<int>::iterator, faiss::IndexHNSWFlat*>> cache_map_; // decouple deserialization to search thread
    std::mutex m_;
    std::unordered_set<int> in_use_indices_;  
    std::unordered_set<int> used_indices_;  // Track sub-HNSWs used in this batch
};


// LRU Cache for sub-HNSW shards with atomic shared_ptr
class LRUCache_atomic {
    friend class LocalHnsw;
    friend class DistributedHnsw;
    friend class PipelinedSearchManager;
public:
    /**
     * Construct an LRUCache that holds up to `capacity` entries.
     */
    explicit LRUCache_atomic(size_t capacity)
        : capacity_(capacity)
    {}

    ~LRUCache_atomic() {
        clear();
    }

    /**
     * Attempt to insert a new entry (serialized bytes) under `key`.  If `key` already
     * existed, its old data is wiped.  If the cache is full, evict_one() is called;
     * if eviction fails, return false.
     *
     * After a successful put(), idx_ptr for this key is guaranteed to be nullptr,
     * so a deserializer may later create a shared_ptr and call update(key, …).
     */
    bool put(int key, std::vector<uint8_t> val) {
        std::lock_guard<std::mutex> lg(m_);

        // If key already exists, remove old entry
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            // Erase old LRU entry
            lru_list_.erase(std::get<1>(it->second));
            // Use atomic operations to match get_index_ptr/update (prevents UB)
            std::atomic_store(&std::get<2>(it->second), std::shared_ptr<faiss::IndexHNSWFlat>(nullptr));
            std::get<0>(it->second)->clear();
            cache_map_.erase(it);
        }

        // If at capacity, evict one
        if (cache_map_.size() >= capacity_) {
            if (!evict_one()) {
                return false;
            }
        }

        // Insert new entry at front of LRU list
        lru_list_.push_front(key);
        cache_map_[key] = { std::make_shared<std::vector<uint8_t>>(std::move(val)), 
                              lru_list_.begin(), 
                              nullptr };
        return true;
    }

    /**
     * Atomically overwrite the shared_ptr<IndexHNSWFlat> for `key`.  If the key does not
     * exist in the cache, this does nothing.
     */
    void update(int key, std::shared_ptr<faiss::IndexHNSWFlat> ptr) {
        std::lock_guard<std::mutex> lg(m_);
        auto it = cache_map_.find(key);
        if (it != cache_map_.end()) {
            // Use atomic_store to replace the shared_ptr<> safely
            std::atomic_store(&std::get<2>(it->second), std::move(ptr));
        }
    }


    std::pair<std::shared_ptr<std::vector<uint8_t>>, std::shared_ptr<faiss::IndexHNSWFlat>>
    get_serialized_and_index(int key) {
        std::lock_guard<std::mutex> lg(m_);
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            return { {}, nullptr };
        }
        // Copy raw_bytes; atomically load the shared_ptr<>
        auto bytes = std::get<0>(it->second);
        auto ptr   = std::atomic_load(&std::get<2>(it->second));
        return { std::move(bytes), std::move(ptr) };
    }

    std::shared_ptr<faiss::IndexHNSWFlat> get_index_ptr(int key) {
        std::lock_guard<std::mutex> lg(m_);
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            return nullptr;
        }
        return std::atomic_load(&std::get<2>(it->second));
    }

    /**
     * Atomically get index pointer AND mark as in-use to prevent eviction.
     * This eliminates the race window between get_index_ptr() and mark_in_use().
     * Returns nullptr if key not found or index not ready.
     */
    std::shared_ptr<faiss::IndexHNSWFlat> get_index_ptr_and_mark_in_use(int key) {
        std::lock_guard<std::mutex> lg(m_);
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) {
            return nullptr;
        }
        auto ptr = std::atomic_load(&std::get<2>(it->second));
        if (ptr != nullptr) {
            in_use_indices_.insert(key);  // Mark as in-use while holding lock
        }
        return ptr;
    }

    /**
     * Return true if we have at least one entry under `key` (serialized and/or index).
     */
    bool has_serialized_or_index(int key) {
        std::lock_guard<std::mutex> lg(m_);
        return cache_map_.count(key) > 0;
    }

    bool has_index(int key) {
        std::lock_guard<std::mutex> lg(m_);
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) return false;
        return (std::atomic_load(&std::get<2>(it->second)) != nullptr);
    }

    bool has_serialized_not_index(int key) {
        std::lock_guard<std::mutex> lg(m_);
        auto it = cache_map_.find(key);
        if (it == cache_map_.end()) return false;
        bool has_bytes = !std::get<0>(it->second)->empty();
        bool no_idx    = (std::atomic_load(&std::get<2>(it->second)) == nullptr);
        return has_bytes && no_idx;
    }

    void mark_in_use(int key) {
        std::lock_guard<std::mutex> lg(m_);
        in_use_indices_.insert(key);
    }

    void unmark_in_use(int key) {
        std::lock_guard<std::mutex> lg(m_);
        in_use_indices_.erase(key);
    }

    void clear_usage_tracking() {
        std::lock_guard<std::mutex> lg(m_);
        used_indices_.clear();
        in_use_indices_.clear();  // Clear both sets (thread-safe)
    }

   void mark_used(int key) {
        std::lock_guard<std::mutex> lg(m_);
        used_indices_.insert(key);
    }

    void reset_used_indices() {
        std::lock_guard<std::mutex> lg(m_);
        used_indices_.clear();
    }

    void clear() {
        std::lock_guard<std::mutex> lg(m_);
        for (auto &kv : cache_map_) {
            // Use atomic_store to match atomic_load/atomic_store used elsewhere
            // Mixing atomic and non-atomic operations on shared_ptr causes UB
            std::atomic_store(&std::get<2>(kv.second), std::shared_ptr<faiss::IndexHNSWFlat>(nullptr));
            std::atomic_store(&std::get<0>(kv.second), std::make_shared<std::vector<uint8_t>>());
        }
        cache_map_.clear();
        lru_list_.clear();
        in_use_indices_.clear();
        used_indices_.clear();
    }


private:
    bool evict_all() {
    // Assumes caller already holds m_ lock
    bool evicted_any = false;
    
    while (true) {
        int victim = -1;
        for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
            int k = *it;
            if (in_use_indices_.count(k) == 0 && used_indices_.count(k) > 0) {
                victim = k;
                break;
            }
        }
        if (victim < 0) {
            // No more victims can be evicted
            break;
        }

        auto map_it = cache_map_.find(victim);
        if (map_it == cache_map_.end()) {
            std::cerr << "LRUCache_atomic CRITICAL INCONSISTENCY: Victim key " << victim 
                    << " found in lru_list_ but not in cache_map_!" << std::endl;
            break; 
        }
        std::list<int>::iterator list_iter_to_erase = std::get<1>(map_it->second);
        lru_list_.erase(list_iter_to_erase);

        // Use atomic operations to match get_index_ptr/update (prevents UB)
        std::atomic_store(&std::get<2>(map_it->second), std::shared_ptr<faiss::IndexHNSWFlat>(nullptr));
        std::get<0>(map_it->second)->clear();
        cache_map_.erase(map_it);
    }
    if(cache_map_.size() < capacity_){
        evicted_any = true;
    }
    return evicted_any;
} 
    bool evict_one() {
        // Assumes caller already holds m_ lock
        if(cache_map_.size() < capacity_) {
            return true;
        }
        int victim = -1;
        for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
            int k = *it;
            if (in_use_indices_.count(k) == 0 && used_indices_.count(k) > 0) {
                victim = k;
                break;
            }
        }
        if (victim < 0) {
            return false;
        }

        auto map_it = cache_map_.find(victim);
        if (map_it == cache_map_.end()) {
            std::cerr << "LRUCache_atomic CRITICAL INCONSISTENCY: Victim key " << victim 
                    << " found in lru_list_ but not in cache_map_!" << std::endl;
            return false; 
        }
        std::list<int>::iterator list_iter_to_erase = std::get<1>(map_it->second);
        lru_list_.erase(list_iter_to_erase);

        // Use atomic operations to match get_index_ptr/update (prevents UB)
        std::atomic_store(&std::get<2>(map_it->second), std::shared_ptr<faiss::IndexHNSWFlat>(nullptr));
        std::get<0>(map_it->second)->clear();
        cache_map_.erase(map_it);
        return true;
    }

    // Capacity limit
    size_t capacity_;

    // Least‐Recently‐Used ordering: front() is most recently used; back() is least.
    std::list<int> lru_list_;

    // Map from key -> ( raw_bytes, LRU iterator, shared_ptr<IndexHNSWFlat> )
    using CacheValue = std::tuple<
        std::shared_ptr<std::vector<uint8_t>>,                  // serialized bytes
        std::list<int>::iterator,             
        std::shared_ptr<faiss::IndexHNSWFlat>  
    >;
    std::unordered_map<int, CacheValue> cache_map_;

    // Protect all of the above
    std::mutex m_;

    // During a search batch, track which keys are currently being touched (in_use)
    // so that they cannot be evicted mid‐search.
    std::unordered_set<int> in_use_indices_;

    // During a search batch, track which keys have been used at least once (used_indices_)
    // to prefer evicting shards that have already served their purpose.
    std::unordered_set<int> used_indices_;
};


class LocalHnsw;
class PipelinedSearchManager { 
friend class LocalHnsw;
friend class LRUCache_atomic;
public:
    PipelinedSearchManager(
        LocalHnsw* parent, 
        int core_start, 
        int cores_per_worker);

    ~PipelinedSearchManager();

    // This method replaces the original function and is now thread-safe.
    std::tuple<double, double, double> process_batch(
        const int n,
        const float* query,
        int K_sub,
        float* distances,
        dhnsw_idx_t* labels,
        std::unordered_map<int, std::unordered_set<int>>& searchset,
        dhnsw_idx_t* sub_hnsw_tags,
        int ef);

private:
    // Worker threads running for the lifetime of the object
    void fetch_worker();
    void deserialize_worker();
    void search_worker();

    // Parent object to access cache and other methods
    LocalHnsw* parent_hnsw_;

    // Threads and synchronization
    std::thread fetch_thread_;
    std::thread deserialize_thread_;
    std::thread search_thread_;
    std::atomic<bool> stop_threads_{false};

    // Communication queues between stages
    ThreadSafeQueue<int> batch_fetch_queue_;
    ThreadSafeQueue<int> batch_deserialize_queue_;
    ThreadSafeQueue<int> batch_ready_queue_;
    
    // Mutex to ensure only one batch is processed at a time
    std::mutex processing_mutex_;
    
public:
    /**
     * Get a lock on processing_mutex_ - for external synchronization.
     * Used by LocalHnsw::init() to wait for in-flight batches to complete
     * before updating offsets and clearing cache.
     */
    std::unique_lock<std::mutex> get_processing_lock() {
        return std::unique_lock<std::mutex>(processing_mutex_);
    }

private:

    // --- Per-Batch State ---
    // These members are set by `process_batch` for each new batch.
    // Access is safe because `processing_mutex_` protects the whole `process_batch` method.
    const float* batch_queries_ptr_ = nullptr;
    float* batch_distances_ptr_ = nullptr;
    dhnsw_idx_t* batch_labels_ptr_ = nullptr;
    dhnsw_idx_t* batch_tags_ptr_ = nullptr;
    std::unordered_map<int, std::unordered_set<int>>* searchset_ptr_ = nullptr;
    int n_ = 0;
    int batch_dim_ = 0;
    int batch_efSearch_ = 0;
    int batch_K_sub_ = 0;

    std::atomic<int> batch_tasks_remaining_{0};
    std::mutex batch_done_mtx_;
    std::condition_variable batch_done_cv_;
    
    std::atomic<double> total_compute_time_{0.0};
    std::atomic<double> total_network_latency_{0.0};
    std::atomic<double> total_deserialize_time_{0.0};
};



class LocalHnsw : public DistributedHnsw { friend class PipelinedSearchManager;
    using idx_t = int64_t; 
    typedef int storage_idx_t;
    private:
        int d; // sychronize by master computing node 
        int num_sub_hnsw;  // sychronize by master computing node
        int meta_M; // sychronize by master computing node
        int sub_M;  // sychronize by master computing node
        
        
        faiss::IndexHNSWFlat* meta_hnsw; // sychronize by master computing node
        std::vector<int> local_sub_hnsw_tag; // assign by master computing node

        /*memory node chop original data into fitted size and send in line. Each computing node construct local_hnsw for their assigned part*/
        
        rdmaio::qp::RC* qp; // RDMA QP for communication
        std::shared_ptr<rdmaio::qp::RC> qp_shared;
        rdmaio::rmem::RegAttr remote_attr; // Server's registered memory region attributes
        rdmaio::Arc<rdmaio::rmem::RegHandler> local_mr; // Client's local memory region
        Arc<rdmaio::rmem::RMem> local_mem_;
        // std::vector<uint64_t> offset; // Offsets for sub_hnsw in server's memory
        std::vector<uint64_t> offset_para_; // Offsets for para in server's memory
        std::vector<uint64_t> offset_subhnsw_; // Offsets for sub_hnsw in server's memory
        std::vector<uint64_t> overflow_; // Offsets for para in local memory
        std::vector<std::vector<dhnsw_idx_t>> mapping; //for easy testing
        
        DhnswClient* dhnsw_client_; // Client to communicate with the server
        
        // Epoch-based RDMA read coordination
        std::atomic<uint64_t> current_epoch_{0};
        std::atomic<uint64_t> current_rdma_base_offset_{0};
        mutable std::shared_mutex epoch_mutex_;  // Reader-writer lock for offset arrays
        std::string client_id_{"local_hnsw"};
        
        // Track active RDMA reads for safe epoch transitions
        std::atomic<int> active_rdma_reads_{0};
        SendTrait* sender_;
        RPCCore<SendTrait, RecvTrait, SManager>* rpc_core_;
        SimpleAllocator* alloc_;
        RegHandler* handler_;
        RecvTrait* recv_transport_;
        uint32_t fetch_rpc_id_;


        CachedOffsets cached_offsets_;

        std::mutex update_mutex_; 


        // Helper methods for offset management
        void init_cached_offsets();

        bool check_and_handle_overflow(int sub_idx, size_t required_size);

        void sync_offsets_to_remote();

        struct InsertionContext {
        int sub_idx;
        size_t n; // original ntotal
        const float* data;
        storage_idx_t entry_point;
        int max_level;
        };

        struct RDMAField {
            std::vector<uint8_t> local_data;
            uint64_t remote_addr;
            size_t size;
        };

        void update_remote_data_with_doorbell(InsertionContext& ctx);

        void update_cached_offsets(InsertionContext& ctx, faiss::IndexHNSWFlat* sub_index);

    public:
        // ---------------------------- pipelined search ----------------------------
        // LRUCache cache_;

        //for search
        LRUCache_atomic cache_;

        //for insert
        std::unordered_map<int, std::shared_ptr<faiss::IndexHNSWFlat>> insert_cache_;
        std::list<int> insert_cache_lru_order_;  // Track LRU order for insert cache
        std::mutex cache_mutex_;
        
        // Overflow detection and reconstruction callback
        using OverflowCallback = std::function<void(int sub_idx, const std::string& overflow_type)>;
        OverflowCallback overflow_callback_;
        std::atomic<bool> overflow_detected_{false};
        std::string last_overflow_type_;
        int last_overflow_sub_idx_{-1};
        
        // Pointer to the PipelinedSearchManager for synchronization during init()
        // This allows init() to wait for in-flight batches to complete before updating state
        PipelinedSearchManager* pipelined_search_manager_ptr_{nullptr};
        
        void set_pipelined_search_manager(PipelinedSearchManager* manager) {
            pipelined_search_manager_ptr_ = manager;
        }
        
        void set_overflow_callback(OverflowCallback callback) {
            overflow_callback_ = callback;
        }
        
        bool has_overflow_detected() const { return overflow_detected_.load(); }
        void clear_overflow_flag() { overflow_detected_.store(false); }
        std::string get_last_overflow_type() const { return last_overflow_type_; }
        int get_last_overflow_sub_idx() const { return last_overflow_sub_idx_; }
        std::tuple<double, double, double> sub_search_pipelined(
            const int n, const float* query, int K_meta, int K_sub, 
            float* distances, dhnsw_idx_t* labels, 
            std::unordered_map<int, std::unordered_set<int>> searchset,
            dhnsw_idx_t* sub_hnsw_tags,
            int ef, fetch_type flag, int core_start, int cores_per_worker);
        std::tuple<double, double, double> sub_search_pipelined_3_stage(
            const int n, const float* query, int K_meta, int K_sub, 
            float* distances, dhnsw_idx_t* labels, 
            std::unordered_map<int, std::unordered_set<int>> searchset,
            dhnsw_idx_t* sub_hnsw_tags,
            int ef, fetch_type flag, int core_start, int cores_per_worker);
        

        std::tuple<double, double, double> sub_search_pipelined_3_stage_init(
            const int n, const float* query, int K_meta, int K_sub, 
            float* distances, dhnsw_idx_t* labels, 
            std::unordered_map<int, std::unordered_set<int>> searchset,
            dhnsw_idx_t* sub_hnsw_tags,
            int ef, fetch_type flag, int core_start, int cores_per_worker); 
        std::unordered_map<int, std::unordered_set<int>> meta_search_pipelined(
            const int n,
            const float* query, 
            int K_meta, 
            float* distances, 
            dhnsw_idx_t* labels, 
            std::vector<int>& sub_hnsw_tosearch,
            int core_start, int cores_per_worker
        );
        std::unordered_map<int, std::unordered_set<int>> meta_search_pipelined_micro(
            const int n,
            const float* query, 
            int K_meta, 
            float* distances, 
            dhnsw_idx_t* labels, 
            std::vector<int>& sub_hnsw_tosearch,
            int core_start, int cores_per_worker
        );
        std::vector<uint8_t> fetch_sub_hnsw_pipelined(int sub_idx);
        std::pair<bool, double> fetch_sub_hnsw_and_put_pipelined(int sub_idx, ThreadSafeQueue<int>& batch_deserialize_queue);
        faiss::IndexHNSWFlat* deserialize_sub_hnsw_pipelined(const std::vector<uint8_t>& data);
        std::pair<bool, double> fetch_sub_hnsw_batch_with_doorbell_and_put_pipelined(const std::vector<int>& sub_indices, ThreadSafeQueue<int>& batch_deserialize_queue, bool& stop);
        // ---------------------------- pipelined search ----------------------------
        LocalHnsw();

        ~LocalHnsw(){
            
            // Clear the cache first to delete all cached sub-indices
            cache_.clear();
            
            // Delete meta_hnsw safely
            if (meta_hnsw) {
                delete meta_hnsw;
                meta_hnsw = nullptr;
            }
            
            // Clean up RPC resources safely  
            if (rpc_core_) {
                delete rpc_core_;
                rpc_core_ = nullptr;
            }
            if (alloc_) {
                delete alloc_;
                alloc_ = nullptr;
            }
            if (recv_transport_) {
                delete recv_transport_;
                recv_transport_ = nullptr;
            }
            
            // Note: dhnsw_client_ is not owned by this class, so we don't delete it
        }

        LocalHnsw(int dim, int sub_partitions, int meta_hnsw_neighbors, int sub_hnsw_neighbors, DhnswClient* client, distance_type flag = Euclidean) 
                : d(dim), num_sub_hnsw(sub_partitions), meta_M(meta_hnsw_neighbors), sub_M(sub_hnsw_neighbors), dhnsw_client_(client), cache_(sub_partitions/20){
            if(flag == Euclidean){
                meta_hnsw = new faiss::IndexHNSWFlat(d, meta_M);
            }
            else if (flag == Angular){
                meta_hnsw = new faiss::IndexHNSWFlat(d, meta_M, faiss::METRIC_INNER_PRODUCT);
            }
            else{
                std::cerr << "Invalid distance type" << std::endl; 
            }
            rdmaio::qp::RC* qp = nullptr;
            DhnswClient* dhnsw_client_ = nullptr; 
            SendTrait* sender_ = nullptr;
            RPCCore<SendTrait, RecvTrait, SManager>* rpc_core_ = nullptr;
            SimpleAllocator* alloc_ = nullptr;
            RegHandler* handler_ = nullptr;
            RecvTrait* recv_transport_ = nullptr;
        }
        
        void init();

        void set_meta_hnsw(faiss::IndexHNSWFlat* meta_hnsw_ptr);

        void set_sub_hnsw_cache(std::vector<faiss::IndexHNSWFlat*> local_sub_hnsw_vec, std::vector<int> local_sub_hnsw_tag);
        
        void set_local_sub_hnsw_tag(std::vector<int> local_sub_hnsw_tag);

        void set_offset_subhnsw(const std::vector<uint64_t>& offset_subhnsw);

        void set_offset_para(const std::vector<uint64_t>& offset_para);

        void set_offset_overflow(const std::vector<uint64_t>& offset_overflow);

        void deserialize_set_sub_hnsw_batch(const std::vector<uint8_t>& data, const std::vector<uint64_t>& offset, std::vector<int> sub_hnsw_tosearch);

        void meta_search(
            const int n,
            const float* query, 
            int K_meta, 
            float* distances, 
            dhnsw_idx_t* labels, 
            std::vector<int>& sub_hnsw_tosearch
        );

        
        void sub_search_each_test(
            const int n, 
            const float* query, 
            int K_meta, 
            int K_sub, 
            float* distances, 
            dhnsw_idx_t* labels, 
            std::vector<int>& sub_hnsw_tosearch, 
            dhnsw_idx_t* sub_hnsw_tags);

        //fetch from server (rdma)
        std::shared_ptr<faiss::IndexHNSWFlat> fetch_sub_hnsw(int sub_idx);

        //fetch from server (grpc)
        faiss::IndexHNSWFlat* fetch_sub_hnsw_grpc(int sub_idx);        

        // Method to remove a sub_hnsw from cache
        int remove_least_recently_used_sub_hnsw();

        void update_lru(int sub_idx);

        void update_local_sub_hnsw(
            std::vector<int> local_sub_hnsw_tag, 
            const std::vector<uint8_t>& data, 
            const std::vector<uint64_t>& offset, 
            std::vector<int> sub_hnsw_tosearch
        );
        
        std::pair<double, double>
        sub_search_each_parallel(
            const int n,
            const float* query,
            int K_meta,
            int K_sub,
            float* distances,
            dhnsw_idx_t* labels,
            std::vector<int>& sub_hnsw_tosearch,
            dhnsw_idx_t* sub_hnsw_tags,
            int ef,
            fetch_type flag);

        // Method to search sub-hnsw, result record in distances and labels (Local view / distributed)
        std::pair<double, double> sub_search_each(
            const int n, const float* query, 
            int K_meta, int K_sub, 
            float* distances, 
            dhnsw_idx_t* labels, 
            std::vector<int>& sub_hnsw_tosearch, 
            dhnsw_idx_t* sub_hnsw_tags,
            int efSearch,
            fetch_type flag = RDMA_DOORBELL
        );

        void sub_search_each_debug(
            const int n, const float* query, 
            int K_meta, int K_sub, 
            float* distances, 
            dhnsw_idx_t* labels, 
            std::vector<int>& sub_hnsw_tosearch, 
            dhnsw_idx_t* sub_hnsw_tags,
            int efSearch
        ); 

        std::pair<double, double> sub_search_each_naive(
            const int n, const float* query, 
            int K_meta, int K_sub, 
            float* distances, 
            dhnsw_idx_t* labels, 
            std::vector<int>& sub_hnsw_tosearch, 
            dhnsw_idx_t* sub_hnsw_tags,
            int efSearch,
            fetch_type flag = RDMA
        );

        void set_rdma_qp(std::shared_ptr<rdmaio::qp::RC> qp, const rmem::RegAttr& remote_attr, const rdmaio::Arc<rdmaio::rmem::RegHandler>& local_mr);

        void set_remote_attr(const rmem::RegAttr& remote_attr);

        void set_local_mr(const rdmaio::Arc<rdmaio::rmem::RegHandler>& local_mr, const Arc<RMem>& local_mem);

        void set_offset(const std::vector<uint64_t>& offset);

        void set_mapping(const std::vector<std::vector<dhnsw_idx_t>>& mapping); 

        int get_original_index(int sub_idx, int local_label);

        std::vector<faiss::IndexHNSWFlat*> fetch_sub_hnsw_batch(const std::vector<int>& sub_indices);

        std::vector<faiss::IndexHNSWFlat*> fetch_sub_hnsw_batch_with_doorbell(const std::vector<int>& sub_indices);
        
        static std::vector<faiss::IndexHNSWFlat*> deserialize_sub_hnsw_batch_with_gap(
            const std::vector<uint8_t>& data, 
            const std::vector<uint64_t>& offset, 
            std::vector<int> sub_hnsw_tosearch
        );

        void rdma_write_to_remote(
            const uint8_t* data, 
            size_t size, 
            size_t remote_offset
        );

        void shift_remote_data(
            size_t start_offset, 
            size_t shift_size
        );

        void update_offsets(
            int sub_idx, 
            size_t shift_size
        );

        void insert_with_record_local(
            const int n, 
            const std::vector<float>& data
        );

        void set_meta_ef_search(int efSearch);
        std::vector<std::vector<dhnsw_idx_t>> get_local_mapping();
        
        void set_rpc_resources(
            UDTransport* sender,
            RPCCore<UDTransport, UDRecvTransport<2048>, UDSessionManager<2048>>* rpc_core,
            RegHandler* handler,
            SimpleAllocator* alloc,
            UDRecvTransport<2048>* recv_transport,
            uint32_t fetch_rpc_id) {
            sender_ = sender;
            rpc_core_ = rpc_core;
            handler_ = handler;
            alloc_ = alloc;
            recv_transport_ = recv_transport;
            fetch_rpc_id_ = fetch_rpc_id;
        }

        // Epoch-based RDMA read coordination methods
        void set_client_id(const std::string& client_id) {
            client_id_ = client_id;
        }

        // Acquire epoch read before batch RDMA operations
        // Returns the base offset for this epoch's data
        bool acquire_epoch_read() {
            if (!dhnsw_client_) return false;
            
            auto epoch_info = dhnsw_client_->AcquireEpochRead(client_id_);
            if (!epoch_info.success) return false;
            
#if DISABLE_RUNTIME_OFFSET_SYNC
            // DEBUG: Offsets are NOT updated here - only epoch tracking is performed.
            // Offsets remain immutable from reconstruction until next reconstruction.
            // This isolates whether segfaults are caused by offset updates during search.
            std::unique_lock<std::shared_mutex> lock(epoch_mutex_);
            
            // Only update epoch counter and RDMA base offset, NOT the offset arrays
            if (epoch_info.epoch != current_epoch_.load()) {
                current_epoch_.store(epoch_info.epoch);
                current_rdma_base_offset_.store(epoch_info.rdma_base_offset);
                // NOTE: offset_subhnsw_, offset_para_, overflow_ are NOT updated
            }
#else
            std::unique_lock<std::shared_mutex> lock(epoch_mutex_);
            
            // Check if epoch changed - if so, we need to update our cached metadata
            if (epoch_info.epoch != current_epoch_.load()) {
                current_epoch_.store(epoch_info.epoch);
                current_rdma_base_offset_.store(epoch_info.rdma_base_offset);
                
                // Update cached offsets with the new epoch's values
                if (epoch_info.has_metadata) {
                    offset_subhnsw_ = epoch_info.offset_subhnsw;
                    offset_para_ = epoch_info.offset_para;
                    overflow_ = epoch_info.overflow;
                }
            }
#endif
            
            active_rdma_reads_.fetch_add(1, std::memory_order_acquire);
            return true;
        }

        // Release epoch read after batch RDMA operations complete
        void release_epoch_read() {
            uint64_t epoch = current_epoch_.load();
            active_rdma_reads_.fetch_sub(1, std::memory_order_release);
            
            if (dhnsw_client_) {
                dhnsw_client_->ReleaseEpochRead(client_id_, epoch);
            }
        }

        // Get the current epoch's RDMA base offset
        uint64_t get_epoch_rdma_base_offset() const {
            return current_rdma_base_offset_.load(std::memory_order_acquire);
        }

        // Get the current epoch number
        uint64_t get_current_epoch() const {
            return current_epoch_.load(std::memory_order_acquire);
        }

        // Thread-safe access to offset arrays (acquires shared lock)
        // Use these when reading offsets from worker threads
        std::shared_lock<std::shared_mutex> get_offsets_read_lock() const {
            return std::shared_lock<std::shared_mutex>(epoch_mutex_);
        }
        
        // Get offset data for a specific sub-index (caller must hold shared lock)
        std::tuple<uint64_t, uint64_t, std::vector<uint64_t>, std::vector<uint64_t>> 
        get_subhnsw_offsets_unsafe(int sub_idx) const {
            uint64_t rel_start = offset_subhnsw_[sub_idx * 2];
            uint64_t rel_end = offset_subhnsw_[sub_idx * 2 + 1];
            
            std::vector<uint64_t> offset_para_tmp;
            for (int i = 0; i < 9; i++) {
                offset_para_tmp.push_back(offset_para_[sub_idx * 9 + i] - rel_start);
            }
            
            std::vector<uint64_t> overflow_tmp;
            size_t overflow_base = sub_idx * 3;
            if (overflow_base + 2 < overflow_.size()) {
                for (int i = 0; i < 3; i++) {
                    overflow_tmp.push_back(overflow_[overflow_base + i] - overflow_[overflow_base]);
                }
            } else {
                overflow_tmp = {0, 0, 0};
            }
            
            return std::make_tuple(rel_start, rel_end, offset_para_tmp, overflow_tmp);
        }

        // Check if we need to re-init due to epoch change
        bool check_epoch_changed() {
            if (!dhnsw_client_) return false;
            auto epoch_info = dhnsw_client_->GetEpochInfo();
            return epoch_info.epoch > current_epoch_.load();
        }

        // Safe epoch-aware re-initialization
        // Waits for active RDMA reads to complete before updating metadata
        void safe_reinit() {
            // Wait for any active RDMA reads to complete
            while (active_rdma_reads_.load(std::memory_order_acquire) > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            
            // Now safe to reinitialize
            init();
        }

        void insert_origin(const int n, const std::vector<float>& data, fetch_type flag);

        void insert_rdma(const int n, const std::vector<float>& data, fetch_type flag, std::vector<int>& sub_hnsw_toinsert);

        faiss::IndexHNSWFlat* fetch_sub_hnsw_for_insert(int sub_idx);
        
        void evict_sub_hnsw_from_insert_cache();
        
        std::shared_ptr<faiss::IndexHNSWFlat> get_from_insert_cache(int sub_idx);
        
        std::pair<double, double> insert_to_server(const int n, std::vector<float>& data); 

        std::vector<faiss::IndexHNSWFlat*> fetch_sub_hnsw_batch_for_insert(const std::vector<int>& sub_indices);

        std::vector<faiss::IndexHNSWFlat*> fetch_sub_hnsw_batch_with_doorbell_for_insert(const std::vector<int>& sub_indices);
      
        faiss::IndexHNSWFlat* deserialize_sub_hnsw_pipelined_(const std::vector<uint8_t>& data, int sub_idx);
        
        void prepare_and_commit_update(int sub_idx, std::shared_ptr<faiss::IndexHNSWFlat> sub_index, const std::vector<float>& insert_data);
    
    
};

struct DataChange {
    uint64_t server_target_offset; 
    const void* local_data_ptr;    
    size_t data_size_bytes; 
    DataChange(uint64_t offset, const void* ptr, size_t size)
        : server_target_offset(offset), local_data_ptr(ptr), data_size_bytes(size) {}       
};

struct OverwriteChange {
    uint64_t server_target_offset;
    std::variant<faiss::Index::idx_t, faiss::storage_idx_t, size_t> value; 
    OverwriteChange(uint64_t offset, std::variant<faiss::Index::idx_t, faiss::storage_idx_t, size_t> val)
        : server_target_offset(offset), value(std::move(val)) {}
};

struct UpdateCommitRecord {
    int sub_index_id;
    // faiss::Index::idx_t ntotal;
    // int max_level;
    // faiss::storage_idx_t entry_point;
    // std::vector<float> xb_append;
    // std::vector<size_t> offsets_append;
    // std::vector<int> levels_append;
    // std::vector<faiss::storage_idx_t> neighbors_append;
    std::vector<OverwriteChange> overwrites;
    std::vector<DataChange> data_changes;

};