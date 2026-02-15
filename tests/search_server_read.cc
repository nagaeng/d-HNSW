// server_read.cc - Complete Hardcoded Read-Only Version

#include <gflags/gflags.h>
#include <grpcpp/grpcpp.h>
#include "../generated/dhnsw.grpc.pb.h"

#include "../../deps/rlib/core/lib.hh"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <fstream>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>
#include <pthread.h>

#include "../dhnsw/DistributedHnsw.h"
#include "../util/read_dataset.h"

// Hardcoded configuration
static const int PORT = 50051;
static const int RDMA_PORT = 8888;
static const int USE_NIC_IDX = 3;
static const int REG_NIC_NAME = 73;
static const int REG_MEM_NAME = 73;
static const std::string INDEX_FILE = "dhnsw_320.idx";
static const std::string META_FILE = "meta_hnsw_320.bin";
static const std::string MAPPING_FILE = "mapping_320.bin";
static const std::string OFFSETS_FILE = "offsets_320.bin";
static const std::string SERVER_IP = "130.127.134.42";

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

// Forward declaration
class DhnswServiceImpl;

void RunGrpcServer(const std::string& server_address,
    const std::vector<uint8_t>& serialized_meta_hnsw,
    const std::vector<size_t>& offset_sub_hnsw,
    const std::vector<size_t>& offset_para,
    const std::vector<size_t>& overflow,
    const std::vector<std::vector<dhnsw_idx_t>>& mapping) {
    
    DhnswServiceImpl service(serialized_meta_hnsw, offset_sub_hnsw, offset_para, overflow, mapping);

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.SetMaxReceiveMessageSize(INT_MAX);
    builder.SetMaxSendMessageSize(INT_MAX);
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

std::vector<uint8_t> load_meta_hnsw(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open meta HNSW file: " << filename << std::endl;
        std::exit(1);
    }
    
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<uint8_t> data(file_size);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    file.close();
    
    std::cout << "✓ Loaded meta HNSW: " << filename << " (" << file_size << " bytes)" << std::endl;
    return data;
}

std::vector<std::vector<dhnsw_idx_t>> load_mapping(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open mapping file: " << filename << std::endl;
        std::exit(1);
    }
    
    size_t mapping_size;
    file.read(reinterpret_cast<char*>(&mapping_size), sizeof(mapping_size));
    
    std::vector<std::vector<dhnsw_idx_t>> mapping(mapping_size);
    
    for (size_t i = 0; i < mapping_size; i++) {
        size_t vec_size;
        file.read(reinterpret_cast<char*>(&vec_size), sizeof(vec_size));
        
        mapping[i].resize(vec_size);
        file.read(reinterpret_cast<char*>(mapping[i].data()), vec_size * sizeof(dhnsw_idx_t));
    }
    
    file.close();
    std::cout << "✓ Loaded mapping: " << filename << " (" << mapping_size << " vectors)" << std::endl;
    return mapping;
}

std::vector<size_t> load_offsets(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open offsets file: " << filename << std::endl;
        std::exit(1);
    }
    
    size_t offset_count;
    file.read(reinterpret_cast<char*>(&offset_count), sizeof(offset_count));
    
    std::vector<size_t> offsets(offset_count);
    file.read(reinterpret_cast<char*>(offsets.data()), offset_count * sizeof(size_t));
    
    file.close();
    std::cout << "✓ Loaded offsets: " << filename << " (" << offset_count << " entries)" << std::endl;
    return offsets;
}

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    bind_thread_to_cores(0, 0, 144);
    omp_set_num_threads(144);
    
    std::cout << "=== DHNSW READ-ONLY SERVER ===" << std::endl;
    std::cout << "Loading pre-built index files..." << std::endl;
    
    // Load all pre-built components
    std::vector<uint8_t> serialized_meta_hnsw = load_meta_hnsw(META_FILE);
    std::vector<std::vector<dhnsw_idx_t>> mapping = load_mapping(MAPPING_FILE);
    std::vector<size_t> offset = load_offsets(OFFSETS_FILE);
    
    std::cout << "\n=== INITIALIZING RDMA ===" << std::endl;
    
    // Initialize RDMA resources
    RCtrl ctrl(RDMA_PORT);
    RDMA_LOG(4) << "RDMA server listens at localhost:" << RDMA_PORT;

    // Open the NIC
    auto nic = RNic::create(RNicInfo::query_dev_names().at(USE_NIC_IDX)).value();
    std::cout << "✓ NIC created" << std::endl;
    RDMA_ASSERT(ctrl.opened_nics.reg(REG_NIC_NAME, nic));

    // Open and map the index file
    int fd = open(INDEX_FILE.c_str(), O_RDONLY);
    if (fd < 0) {
        perror("Error opening index file");
        std::exit(1);
    }
    
    struct stat st;
    if (fstat(fd, &st) != 0) {
        perror("Error getting file stats");
        close(fd);
        std::exit(1);
    }
    
    size_t total_size = st.st_size;
    std::cout << "✓ Index file size: " << total_size << " bytes" << std::endl;
 
    auto mr_memory = Arc<RMem>(new RMem(total_size));
    RDMA_ASSERT(ctrl.registered_mrs.create_then_reg(REG_MEM_NAME, mr_memory, nic));

    auto mr_attr = ctrl.registered_mrs.query(REG_MEM_NAME).value()->get_reg_attr().value();
    std::cout << "✓ RDMA memory region created" << std::endl;
    uint8_t* reg_mem = reinterpret_cast<uint8_t*>(mr_attr.buf);
    
    // Load the index file into RDMA memory
    std::cout << "Loading index into RDMA buffer..." << std::endl;
    size_t bytes_left = total_size;
    size_t offset_file = 0;
    while (bytes_left) {
        ssize_t got = ::pread(fd, reg_mem + offset_file, bytes_left, offset_file);
        if (got <= 0) {
            perror("Error reading index file");
            close(fd);
            std::exit(1);
        }
        bytes_left -= got;
        offset_file += got;
    }
    close(fd);
    std::cout << "✓ Loaded " << offset_file << " bytes into RDMA buffer" << std::endl;
    
    ctrl.start_daemon();
    std::cout << "✓ RDMA key: " << mr_attr.key << std::endl;
    RDMA_LOG(4) << "RDMA resources initialized";

    // Start the gRPC server
    std::string server_address = SERVER_IP + ":" + std::to_string(PORT);
    
    // Empty vectors for unused parameters
    const std::vector<size_t> offset_para;
    const std::vector<size_t> overflow;
    
    std::thread grpc_server_thread(RunGrpcServer, server_address, serialized_meta_hnsw, 
                                   offset, offset_para, overflow, mapping);

    std::cout << "\n=== SERVER READY ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  • gRPC server: " << server_address << std::endl;
    std::cout << "  • RDMA server: localhost:" << RDMA_PORT << std::endl;
    std::cout << "  • RDMA key: " << mr_attr.key << std::endl;
    std::cout << "\nLoaded components:" << std::endl;
    std::cout << "  • Meta HNSW: " << serialized_meta_hnsw.size() << " bytes" << std::endl;
    std::cout << "  • Mapping: " << mapping.size() << " vectors" << std::endl;
    std::cout << "  • Offsets: " << offset.size() << " entries" << std::endl;
    std::cout << "  • Index: " << total_size << " bytes in RDMA" << std::endl;
    std::cout << "\nServer is ready to handle queries!" << std::endl;
    std::cout << "====================" << std::endl;
    
    // Wait for the server
    grpc_server_thread.join();

    return 0;
}