#pragma once
#include <iostream>
#include <vector>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/MetaIndexes.h>
#include <faiss/impl/HNSW.h>
#include <faiss/Clustering.h>
#include <sstream>
#include <omp.h>
#include <faiss/utils/Heap.h>
#include <list>
#include <cstdint> // for uint8_t
#include "../../deps/faiss/impl/io.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
struct DirectMemoryIOReader : faiss::IOReader {
    const uint8_t* const start_ptr; 
    const uint8_t* current_ptr;
    const uint8_t* const end_ptr;
    const size_t total_size;      

    DirectMemoryIOReader(const std::vector<uint8_t>& source_data)
        : start_ptr(source_data.data()),
          current_ptr(source_data.data()),
          end_ptr(source_data.data() + source_data.size()),
          total_size(source_data.size()) {
        name = "<DirectMemoryIOReaderVec>";
    }

    DirectMemoryIOReader(const uint8_t* ptr, size_t size)
        : start_ptr(ptr),
          current_ptr(ptr),
          end_ptr(ptr + size),
          total_size(size) {
        name = "<DirectMemoryIOReaderPtr>";
    }

    size_t operator()(void* ptr_out, size_t item_size, size_t n_items) override {
        if (item_size == 0 || n_items == 0) {
            return 0;
        }
        size_t bytes_requested = item_size * n_items;
        size_t bytes_available = (current_ptr < end_ptr) ? (size_t)(end_ptr - current_ptr) : 0;
        size_t bytes_to_copy = std::min(bytes_requested, bytes_available);

        if (bytes_to_copy > 0) {
            std::memcpy(ptr_out, current_ptr, bytes_to_copy);
            current_ptr += bytes_to_copy;
        }

        return bytes_to_copy / item_size;
    }


    size_t tell() {
        return current_ptr - start_ptr;
    }
};
namespace faiss{
    typedef int storage_idx_t;
    static void write_dhnsw_index_header(
        const Index* idx, 
        IOWriter* f, 
        std::vector<uint64_t>& offset_para, 
        int sub_idx, size_t& current_pos);


    void write_dhnsw_index_init(
        const Index* idx, 
        IOWriter* f, 
        std::vector<uint64_t>& offset_para, 
        int sub_idx, 
        size_t& current_pos);

    void write_dhnsw_index_init_(
        const Index* idx, 
        IOWriter* f, 
        std::vector<uint64_t>& offset_para, 
        int sub_idx, 
        size_t& current_pos);
    
    static void write_original_index_header (
        const Index *idx, 
        IOWriter *f,
        size_t& current_pos,
        size_t& ptr_storage_ntotal);

    static void write_dhnsw_HNSW(
        const HNSW* hnsw, 
        IOWriter* f, 
        std::vector<uint64_t>& offset_para, 
        int sub_idx, 
        size_t& current_pos);
    
    static void write_dhnsw_HNSW_(
        const HNSW* hnsw, 
        IOWriter* f, 
        std::vector<uint64_t>& offset_para, 
        int sub_idx, 
        size_t& current_pos);

    static void write_dhnsw_storage(
        const Index* idx, 
        IOWriter* f, 
        std::vector<uint64_t>& offset_para, 
        int sub_idx, 
        size_t& current_pos);

    static void write_dhnsw_storage_(
        const Index* idx, 
        IOWriter* f, 
        std::vector<uint64_t>& offset_para, 
        int sub_idx, 
        size_t& current_pos);
    static void read_dhnsw_storage(
        IndexFlat* storage,
        IOReader* f,
        std::vector<uint64_t>& offset_para,
        int sub_idx,
        size_t& current_pos);
    static void read_dhnsw_HNSW(
        HNSW* hnsw,
        IOReader* f,
        std::vector<uint64_t>& offset_para,
        int sub_idx,
        size_t& current_pos);
    static void read_dhnsw_HNSW_(
        HNSW* hnsw,
        IOReader* f,
        std::vector<uint64_t>& offset_para,
        int sub_idx,
        size_t& current_pos);
    static void read_dhnsw_storage(
        Index* idx,
        IOReader* f,
        std::vector<uint64_t>& offset_para,
        int sub_idx,
        size_t& current_pos);
    static void read_dhnsw_storage_(
        Index* idx,
        IOReader* f,
        std::vector<uint64_t>& offset_para,
        int sub_idx,
        size_t& current_pos);

    void read_dhnsw_index_init(
        Index* idx,
        IOReader* f,
        std::vector<uint64_t>& offset_para,
        int sub_idx,
        size_t& current_pos);

    void read_dhnsw_index_init_(
        Index* idx,
        IOReader* f,
        std::vector<uint64_t>& offset_para,
        int sub_idx,
        size_t& current_pos);

    void seek_in_reader(
        faiss::VectorIOReader& reader, 
        size_t pos);
    void seek_in_reader_(
        faiss::VectorIOReader& reader, 
        size_t pos);
    void seek_in_reader(
        DirectMemoryIOReader& reader, 
        size_t pos);
    size_t get_reader_position(
        const faiss::VectorIOReader& reader);
    void seek_in_generic_reader(IOReader* f, size_t pos);
    size_t get_generic_reader_position(IOReader* f);
    void read_dhnsw_single_sub_hnsw(
        IndexHNSW* idx,
        const std::vector<uint8_t>& data,
        std::vector<uint64_t>& offset_para,
        std::vector<uint64_t>& offset_sub_hnsw,
        int sub_idx);
    void update_current_pos(IOWriter* f, size_t & current_pos);

    void read_HNSW_optimized(faiss::HNSW* hnsw, faiss::IOReader* f);
    void read_HNSW_optimized_(faiss::HNSW* hnsw, faiss::IOReader* f,
    std::vector<uint64_t>& offset_para, 
    std::vector<uint64_t>& overflow);
    faiss::IndexHNSWFlat* read_index_HNSWFlat_optimized(faiss::IOReader* f);
    faiss::IndexHNSWFlat* read_index_HNSWFlat_optimized_(
        faiss::IOReader* f, 
        std::vector<uint64_t>& offset_para, 
        std::vector<uint64_t>& overflow);
    faiss::Index* read_storage_optimized(faiss::IOReader* f);
    faiss::Index* read_storage_optimized_(faiss::IOReader* f,
    std::vector<uint64_t>& offset_para, 
    std::vector<uint64_t>& overflow);
    void fast_read_exact(faiss::IOReader* f, void* dst, size_t item_size, size_t item_count);
}

class EfficientMMapBvecs {
private:
    int fd;
    uint8_t* raw_mapped_data;
    size_t file_size;
    int dimension;
    int num_vectors;
    int max_vectors;
    int vec_size;
    
public:
    EfficientMMapBvecs(const std::string& filename, int max_vectors_limit = -1) 
        : fd(-1), raw_mapped_data(nullptr), max_vectors(max_vectors_limit) {
        
        std::cout << "Memory mapping bvecs file: " << filename << std::endl;
        if (max_vectors_limit > 0) {
            std::cout << "Limiting to first " << max_vectors_limit << " vectors" << std::endl;
        }
        
        // 打开文件
        fd = open(filename.c_str(), O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("Cannot open file: " + filename);
        }
        
        // 获取文件大小
        struct stat st;
        if (fstat(fd, &st) != 0) {
            close(fd);
            throw std::runtime_error("Cannot stat file: " + filename);
        }
        file_size = st.st_size;
        
        // 读取维度（从文件开头）
        uint8_t temp_buffer[4];
        if (pread(fd, temp_buffer, sizeof(int), 0) != sizeof(int)) {
            close(fd);
            throw std::runtime_error("Cannot read dimension from file");
        }
        dimension = *reinterpret_cast<int*>(temp_buffer);
        
        // 计算每个向量的大小和总向量数
        vec_size = sizeof(int) + dimension * sizeof(uint8_t);
        int total_vectors_in_file = file_size / vec_size;
        
        // 应用向量数限制
        if (max_vectors > 0 && max_vectors < total_vectors_in_file) {
            num_vectors = max_vectors;
        } else {
            num_vectors = total_vectors_in_file;
        }
        
        std::cout << "Dimension: " << dimension << std::endl;
        std::cout << "Total vectors in file: " << total_vectors_in_file << std::endl;
        std::cout << "Vectors to process: " << num_vectors << std::endl;
        
        // 计算需要映射的文件大小（只映射需要的部分）
        size_t mapped_file_size = static_cast<size_t>(num_vectors) * vec_size;
        std::cout << "Mapping " << mapped_file_size / (1024*1024*1024) << " GB of file data" << std::endl;
        
        // 内存映射需要的文件部分
        raw_mapped_data = (uint8_t*)mmap(nullptr, mapped_file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (raw_mapped_data == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("Cannot memory map file: " + filename);
        }
        
        std::cout << "Memory mapping completed successfully!" << std::endl;
    }
    
    ~EfficientMMapBvecs() {
        if (raw_mapped_data != nullptr) {
            size_t mapped_file_size = static_cast<size_t>(num_vectors) * vec_size;
            munmap(raw_mapped_data, mapped_file_size);
        }
        if (fd != -1) {
            close(fd);
        }
    }
    
    // 批量转换指定范围的向量
    std::vector<float> get_vectors_as_float(int start_idx, int count) const {
        if (start_idx < 0 || start_idx + count > num_vectors) {
            throw std::out_of_range("Vector range out of bounds");
        }
        
        std::vector<float> result;
        result.reserve(count * dimension);
        
        for (int i = 0; i < count; i++) {
            uint8_t* vec_ptr = raw_mapped_data + (start_idx + i) * vec_size;
            
            // 验证维度字段
            int dim = *reinterpret_cast<int*>(vec_ptr);
            if (dim != dimension) {
                throw std::runtime_error("Dimension mismatch at vector " + std::to_string(start_idx + i));
            }
            
            uint8_t* data_ptr = vec_ptr + sizeof(int);
            
            // 转换 uint8_t 到 float
            for (int j = 0; j < dimension; j++) {
                result.push_back(static_cast<float>(data_ptr[j]));
            }
        }
        
        return result;
    }
    
    // 分批转换所有向量为 std::vector<float>
    std::vector<float> to_vector_batched(int batch_size = 1000000) const {
        std::cout << "Converting all vectors to float in batches of " << batch_size << std::endl;
        
        std::vector<float> result;
        result.reserve(static_cast<size_t>(num_vectors) * dimension);
        
        for (int start = 0; start < num_vectors; start += batch_size) {
            int count = std::min(batch_size, num_vectors - start);
            
            std::cout << "Converting batch: " << start << " to " << (start + count - 1) << std::endl;
            
            auto batch_data = get_vectors_as_float(start, count);
            result.insert(result.end(), batch_data.begin(), batch_data.end());
            
            // 清理批次数据以释放内存
            batch_data.clear();
            batch_data.shrink_to_fit();
        }
        
        std::cout << "All vectors converted successfully!" << std::endl;
        return result;
    }
    
    // 获取单个向量
    std::vector<float> get_vector(int index) const {
        return get_vectors_as_float(index, 1);
    }
    
    // 获取原始数据指针（用于高级用法）
    const uint8_t* get_raw_vector_data(int index) const {
        if (index < 0 || index >= num_vectors) {
            throw std::out_of_range("Vector index out of range");
        }
        return raw_mapped_data + index * vec_size + sizeof(int);
    }
    
    // 获取基本信息
    int get_dimension() const { return dimension; }
    int get_num_vectors() const { return num_vectors; }
    size_t get_total_elements() const { return static_cast<size_t>(num_vectors) * dimension; }
};

class VectorAdapter {
private:
    const float* data_ptr;
    size_t data_size;
    
public:
    VectorAdapter(const float* ptr, size_t size) : data_ptr(ptr), data_size(size) {}
    
    const float* data() const { return data_ptr; }
    size_t size() const { return data_size; }

    const float& operator[](size_t index) const { return data_ptr[index]; }
};

