#include "dhnsw_io.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/MetaIndexes.h>
#include <faiss/impl/HNSW.h>
#include <faiss/Clustering.h>
#include <sstream>
// #include <KaHIP/interface/kaHIP_interface.h>
#include <omp.h>
#include <faiss/utils/Heap.h>
#include <shared_mutex>
#include <faiss/index_io.h>
#include <faiss/impl/io_macros.h>
#include <faiss/index_io.h>

#include <faiss/impl/io.h> 
#include <cstdint> // for uint8_t
#include <sys/mman.h>

namespace faiss{

    void update_current_pos(IOWriter* f, size_t & current_pos) {
        VectorIOWriter* vw = dynamic_cast<VectorIOWriter*>(f);
        if (vw) {
            current_pos = vw->data.size();
        }
    }

    static void write_dhnsw_HNSW(const HNSW* hnsw, IOWriter* f, std::vector<uint64_t>& offset_para, int sub_idx, size_t& current_pos) {
        size_t base_idx = sub_idx * 15;
        
        // Write and record levels with gap
        offset_para[base_idx + 1] = current_pos; // levels start
        
        WRITEVECTOR(hnsw->levels);
        update_current_pos(f, current_pos);
        // Add 20% gap for levels
        size_t levels_gap = (hnsw->levels.size() * sizeof(hnsw->levels[0])) / 5;
        std::vector<char> gap(levels_gap, 0);
        f->operator()(gap.data(), 1, levels_gap);
        update_current_pos(f, current_pos);
       
        // Write and record offsets with gap
        offset_para[base_idx + 2] = current_pos; // offsets start
        WRITEVECTOR(hnsw->offsets);
        update_current_pos(f, current_pos);
        // Add 20% gap for offsets
        size_t offsets_gap = (hnsw->offsets.size() * sizeof(hnsw->offsets[0])) / 5;
        gap.resize(offsets_gap, 0);
        f->operator()(gap.data(), 1, offsets_gap);
        update_current_pos(f, current_pos);

        // Write transposed neighbors
        // Write neighbors header
        size_t total_nodes = hnsw->levels.size();
        size_t num_levels = hnsw->cum_nneighbor_per_level.size() - 1;
        size_t total_size = 0;
        for (size_t level = 0; level < num_levels; level++) {
            total_size += total_nodes * hnsw->nb_neighbors(level);
        }

        offset_para[base_idx + 3] = current_pos; // neighbors start, including header (total_size)
        WRITE1(total_size);
        update_current_pos(f, current_pos);
        // Write neighbors in transposed format, level by level
        if(hnsw->max_level < 2) {
            // std::cout << "hnsw->max_level" << hnsw->max_level << std::endl;
            // std::cout << "sub_idx" << sub_idx << std::endl;
            std::cerr << "max level is too small" << std::endl;
            exit(1);
        }
        if(hnsw->max_level == 2){
            // std::cout << "sub_idx" << sub_idx << std::endl; 
        }
        bool bottom_marker_set = false;
        for (int level = hnsw->max_level; level >= 0; level--) {
            
            if (level == 1 && hnsw->max_level >= 2 && !bottom_marker_set) {
                offset_para[base_idx + 13] = current_pos; // mark bottom two layers start
                bottom_marker_set = true;
                // std::cout << "write_dhnsw_HNSW bottom two layers neighbors start" << std::endl;
            }

                for (int node = 0; node < static_cast<int>(total_nodes); node++) {
                    if (level < hnsw->levels[node]) {
                        int nbrs = hnsw->nb_neighbors(level);
                        size_t offset_start = hnsw->offsets[node];
                        size_t cum_neighbors_before = hnsw->cum_nb_neighbors(level);
                            for (int pos = 0; pos < nbrs; pos++) {
                                size_t idx = offset_start + cum_neighbors_before + pos;
                                WRITE1(hnsw->neighbors[idx]);
                                // if(level > 0) {
                                //     // std::cout << "write_dhnsw_HNSW neighbor for node " << node 
                                //             << " level " << level << " pos " << pos << std::endl;
                                // }
                        }
                }
            }
        }
        
        update_current_pos(f, current_pos);
        offset_para[base_idx + 14] = current_pos; // 2 bottom layers neighbors end (TODO: May optimize)
        // Add neighbors gap
        size_t neighbors_gap = total_size * sizeof(hnsw->neighbors[0]) / 5;
        gap.resize(neighbors_gap, 0);
        f->operator()(gap.data(), 1, neighbors_gap);
        update_current_pos(f, current_pos);
        // std::cout << "write_dhnsw_HNSW neighbors gap done" << std::endl;
        // Write remaining HNSW data
        offset_para[base_idx + 5] = current_pos; // entry_point position
        WRITE1(hnsw->entry_point);
        update_current_pos(f, current_pos);
        offset_para[base_idx + 6] = current_pos; // max_level position
        WRITE1(hnsw->max_level);
        update_current_pos(f, current_pos);
        WRITE1(hnsw->efConstruction);
        update_current_pos(f, current_pos);
        WRITE1(hnsw->efSearch);
        update_current_pos(f, current_pos);
        WRITE1(hnsw->upper_beam);
        update_current_pos(f, current_pos);
        // std::cout << "write_dhnsw_HNSW done" << std::endl;
    } 


    static void write_dhnsw_HNSW_(const HNSW* hnsw, IOWriter* f, std::vector<uint64_t>& offset_para, int sub_idx, size_t& current_pos) {
        size_t base_idx = sub_idx * 9;
        
        // Write and record levels with gap
        offset_para[base_idx + 1] = current_pos; // levels start
        
        WRITEVECTOR(hnsw->levels);
        update_current_pos(f, current_pos);
        // Add 10% gap for levels
        size_t levels_gap = (hnsw->levels.size() * sizeof(hnsw->levels[0])) / 10;
        std::vector<char> gap(levels_gap, 0);
        f->operator()(gap.data(), 1, levels_gap);
        update_current_pos(f, current_pos);
       
        // Write and record offsets with gap
        offset_para[base_idx + 2] = current_pos; // offsets start
        WRITEVECTOR(hnsw->offsets);
        update_current_pos(f, current_pos);
        // Add 10% gap for offsets
        size_t offsets_gap = (hnsw->offsets.size() * sizeof(hnsw->offsets[0])) / 10;
        gap.resize(offsets_gap, 0);
        f->operator()(gap.data(), 1, offsets_gap);
        update_current_pos(f, current_pos);
        // Write and record neighbors with gap
        offset_para[base_idx + 3] = current_pos; // neighbors start
        WRITEVECTOR(hnsw->neighbors);
        update_current_pos(f, current_pos);
        // Add neighbors gap
        size_t neighbors_gap = hnsw->neighbors.size() * sizeof(hnsw->neighbors[0]) / 10;
        gap.resize(neighbors_gap, 0);
        f->operator()(gap.data(), 1, neighbors_gap);
        update_current_pos(f, current_pos);
        // std::cout << "write_dhnsw_HNSW neighbors gap done" << std::endl;
        // Write remaining HNSW data
        offset_para[base_idx + 4] = current_pos; // entry_point position
        WRITE1(hnsw->entry_point);
        update_current_pos(f, current_pos);
        offset_para[base_idx + 5] = current_pos; // max_level position
        WRITE1(hnsw->max_level);
        update_current_pos(f, current_pos);
        WRITE1(hnsw->efConstruction);
        update_current_pos(f, current_pos);
        WRITE1(hnsw->efSearch);
        update_current_pos(f, current_pos);
        WRITE1(hnsw->upper_beam);
        update_current_pos(f, current_pos);
        // std::cout << "write_dhnsw_HNSW done" << std::endl;
    } 
    static void write_dhnsw_index_header(const Index* idx, IOWriter* f, std::vector<uint64_t>& offset_para, int sub_idx, size_t& current_pos) {
        // std::cout << "write_dhnsw_index_header" << std::endl;
        WRITE1(idx->d);
        update_current_pos(f, current_pos);
        
        offset_para[sub_idx * 15] = current_pos; // Record ntotal position
        WRITE1(idx->ntotal);
        update_current_pos(f, current_pos);
        
        Index::idx_t dummy = 1 << 20;
        WRITE1(dummy);
        update_current_pos(f, current_pos);
        WRITE1(dummy);
        update_current_pos(f, current_pos);
        WRITE1(idx->is_trained);
        update_current_pos(f, current_pos);
        WRITE1(idx->metric_type);
        update_current_pos(f, current_pos);
        if (idx->metric_type > 1) {
            WRITE1(idx->metric_arg);
            update_current_pos(f, current_pos);
        }
        // std::cout << "write_dhnsw_index_header done" << std::endl;
    }
    static void write_dhnsw_index_header_(const Index* idx, IOWriter* f, std::vector<uint64_t>& offset_para, int sub_idx, size_t& current_pos) {
        // std::cout << "write_dhnsw_index_header" << std::endl;
        WRITE1(idx->d);
        update_current_pos(f, current_pos);
        
        offset_para[sub_idx * 9] = current_pos; // Record ntotal position
        WRITE1(idx->ntotal);
        update_current_pos(f, current_pos);
        
        Index::idx_t dummy = 1 << 20;
        WRITE1(dummy);
        update_current_pos(f, current_pos);
        WRITE1(dummy);
        update_current_pos(f, current_pos);
        WRITE1(idx->is_trained);
        update_current_pos(f, current_pos);
        WRITE1(idx->metric_type);
        update_current_pos(f, current_pos);
        if (idx->metric_type > 1) {
            WRITE1(idx->metric_arg);
            update_current_pos(f, current_pos);
        }
        // std::cout << "write_dhnsw_index_header done" << std::endl;
    }
    static void write_dhnsw_storage(const Index* idx, IOWriter* f, std::vector<uint64_t>& offset_para, int sub_idx, size_t& current_pos) {
        // std::cout << "write_dhnsw_storage" << std::endl;
        const IndexFlat* idxf = dynamic_cast<const IndexFlat*>(idx);
        FAISS_THROW_IF_NOT(idxf);

        uint32_t h = fourcc(
            idxf->metric_type == METRIC_INNER_PRODUCT ? "IxFI" :
            idxf->metric_type == METRIC_L2 ? "IxF2" : "IxFl");
        WRITE1(h);
        update_current_pos(f, current_pos);

        // Write header information
        write_original_index_header(idxf, f, current_pos, offset_para[sub_idx * 15 + 7]);
        //update_current_pos(f, current_pos);
        
        // Write vector data with gap
        offset_para[sub_idx * 15 + 8] = current_pos; // xb start position
        WRITEVECTOR(idxf->xb);
        update_current_pos(f, current_pos);
        // Add 20% gap for xb
        size_t xb_gap = (idxf->xb.size() * sizeof(float)) / 5;
        std::vector<char> gap(xb_gap, 0);
        f->operator()(gap.data(), 1, xb_gap);
        update_current_pos(f, current_pos);
        offset_para[sub_idx * 15 + 12] = current_pos; // xb max boundary
        // std::cout << "write_dhnsw_storage done" << std::endl;
    }
    static void write_dhnsw_storage_(const Index* idx, IOWriter* f, std::vector<uint64_t>& offset_para, int sub_idx, size_t& current_pos) {
        // std::cout << "write_dhnsw_storage" << std::endl;
        const IndexFlat* idxf = dynamic_cast<const IndexFlat*>(idx);
        FAISS_THROW_IF_NOT(idxf);

        uint32_t h = fourcc(
            idxf->metric_type == METRIC_INNER_PRODUCT ? "IxFI" :
            idxf->metric_type == METRIC_L2 ? "IxF2" : "IxFl");
        WRITE1(h);
        update_current_pos(f, current_pos);

        // Write header information
        write_original_index_header(idxf, f, current_pos, offset_para[sub_idx * 9 + 6]);
        //update_current_pos(f, current_pos);
        
        // Write vector data with gap
        offset_para[sub_idx * 9 + 7] = current_pos; // xb start position
        WRITEVECTOR(idxf->xb);
        update_current_pos(f, current_pos);
        // Add 10% gap for xb
        size_t xb_gap = (idxf->xb.size() * sizeof(float)) / 10;
        std::vector<char> gap(xb_gap, 0);
        f->operator()(gap.data(), 1, xb_gap);
        update_current_pos(f, current_pos);
        offset_para[sub_idx * 9 + 8] = current_pos; // xb max boundary
        // std::cout << "write_dhnsw_storage done" << std::endl;
    }
    void write_dhnsw_index_init(const Index* idx, IOWriter* f, std::vector<uint64_t>& offset_para, int sub_idx, size_t& current_pos) {
        // std::cout << "write_dhnsw_index_init" << std::endl;
        const IndexHNSW* idxhnsw = dynamic_cast<const IndexHNSW*>(idx);
        FAISS_THROW_IF_NOT(idxhnsw);
        
        uint32_t h = fourcc("IHNf");
        FAISS_THROW_IF_NOT(h != 0);
        WRITE1(h);
        update_current_pos(f, current_pos);
        
        write_dhnsw_index_header(idxhnsw, f, offset_para, sub_idx, current_pos);
        write_dhnsw_HNSW(&idxhnsw->hnsw, f, offset_para, sub_idx, current_pos);
        write_dhnsw_storage(idxhnsw->storage, f, offset_para, sub_idx, current_pos);
        // std::cout << "write_dhnsw_index_init done" << std::endl;
    }
    void write_dhnsw_index_init_(const Index* idx, IOWriter* f, std::vector<uint64_t>& offset_para, int sub_idx, size_t& current_pos) {
        // std::cout << "write_dhnsw_index_init_" << std::endl;
        const IndexHNSW* idxhnsw = dynamic_cast<const IndexHNSW*>(idx);
        FAISS_THROW_IF_NOT(idxhnsw);
        
        uint32_t h = fourcc("IHNf");
        FAISS_THROW_IF_NOT(h != 0);
        WRITE1(h);
        update_current_pos(f, current_pos);
        
        write_dhnsw_index_header_(idxhnsw, f, offset_para, sub_idx, current_pos);
        write_dhnsw_HNSW_(&idxhnsw->hnsw, f, offset_para, sub_idx, current_pos);
        write_dhnsw_storage_(idxhnsw->storage, f, offset_para, sub_idx, current_pos);
        // std::cout << "write_dhnsw_index_init done" << std::endl;
    }

    static void write_original_index_header (const Index *idx, IOWriter *f, size_t& current_pos, size_t& ptr_storage_ntotal) {
        // std::cout << "write_original_index_header" << std::endl;
        WRITE1 (idx->d);
        update_current_pos(f, current_pos);
        ptr_storage_ntotal = current_pos; // storage ntotal position
        WRITE1 (idx->ntotal);
        update_current_pos(f, current_pos);
        Index::idx_t dummy = 1 << 20;
        WRITE1 (dummy);
        WRITE1 (dummy);
        WRITE1 (idx->is_trained);
        WRITE1 (idx->metric_type);
        if (idx->metric_type > 1) {
            WRITE1 (idx->metric_arg);
        }
        update_current_pos(f, current_pos);
        // std::cout << "write_original_index_header done" << std::endl;
    }

    static void read_original_index_header(Index* idx, IOReader* f) {

        // std::cout << "Reading original index header 2..." << std::endl;

        if (!idx) {
            throw std::runtime_error("Index pointer is null");
        } 
        
        READ1(idx->d);
        // std::cout << "raed d"  << idx->d << std::endl;
        READ1(idx->ntotal);
        // std::cout << "raed ntotal"  << idx->ntotal << std::endl;
        Index::idx_t dummy;
        READ1(dummy);
        // std::cout << "raed dummy"  << dummy << std::endl;
        READ1(dummy);
        // std::cout << "raed dummy"  << dummy << std::endl;
        READ1(idx->is_trained);
        // std::cout << "raed is_trained"  << idx->is_trained << std::endl;
        READ1(idx->metric_type);
        // std::cout << "raed metric_type"  << idx->metric_type << std::endl;
        if (idx->metric_type > 1) {
            READ1(idx->metric_arg);
        }
        // std::cout << "read_original_index_header done" << std::endl;
    }

    static void read_dhnsw_index_header(
        Index* idx, 
        IOReader* f,
        std::vector<uint64_t>& offset_para,
        int sub_idx,
        size_t& current_pos) {
        // std::cout << "Reading original index header.1.." << std::endl;
        read_original_index_header(idx, f);
        current_pos = get_generic_reader_position(f); 
    }

    static void read_dhnsw_HNSW(
        HNSW* hnsw,
        IOReader* f,
        std::vector<uint64_t>& offset_para,
        int sub_idx,
        size_t& current_pos) {
    // std::cout << "read_dhnsw_HNSW" << std::endl;
    size_t base_idx = sub_idx * 15;
    
    // Read levels
    current_pos = offset_para[base_idx + 1];
    seek_in_reader(dynamic_cast<faiss::VectorIOReader&>(*f), current_pos);
    READVECTOR(hnsw->levels);
    
    // Read offsets
    current_pos = offset_para[base_idx + 2];
    seek_in_reader(dynamic_cast<faiss::VectorIOReader&>(*f), current_pos);
    READVECTOR(hnsw->offsets);
    
    // Read neighbors
    current_pos = offset_para[base_idx + 3];
    seek_in_reader(dynamic_cast<faiss::VectorIOReader&>(*f), current_pos);
  
    // Read total size first (header)
    size_t total_size;
    READ1(total_size);
    hnsw->neighbors.resize(total_size);

    // Calculate dimensions
    size_t total_nodes = hnsw->levels.size();
    size_t num_levels = hnsw->cum_nneighbor_per_level.size() - 1;
    // Read and reconstruct neighbors
    for (int level = hnsw->max_level; level >= 0; level--) {
        int nbrs = hnsw->nb_neighbors(level);
        for (int pos = 0; pos < nbrs; pos++) {
            for (int node = 0; node < static_cast<int>(total_nodes); node++) {
                if(level < hnsw->levels[node]) {
                int nbrs = hnsw->nb_neighbors(level);
                    for (int pos = 0; pos < nbrs; pos++) { 
                        size_t offset_start = hnsw->offsets[node];
                        size_t cum_neighbors_before = hnsw->cum_nb_neighbors(level);
                        size_t idx = offset_start + cum_neighbors_before + pos;
                        if (idx >= hnsw->neighbors.size()) {
                            throw std::runtime_error("Neighbor index out of bounds");
                        }
                        storage_idx_t neighbor;
                        READ1(neighbor);
                        current_pos += sizeof(storage_idx_t);
                        hnsw->neighbors[idx] = neighbor;
                    }
                }
            }
        }
    }
    current_pos = offset_para[base_idx + 5];
    
    // Read remaining HNSW data
    seek_in_reader(dynamic_cast<faiss::VectorIOReader&>(*f), current_pos);
    READ1(hnsw->entry_point);
    // std::cout << "read entry_point" << hnsw->entry_point << std::endl;
    
    READ1(hnsw->max_level);
    // std::cout << "read max_level" << hnsw->max_level << std::endl;
    READ1(hnsw->efConstruction);
    // std::cout << "read efConstruction" << hnsw->efConstruction << std::endl;
    READ1(hnsw->efSearch);
    // std::cout << "read efSearch" << hnsw->efSearch << std::endl;
    READ1(hnsw->upper_beam);
    // std::cout << "read upper_beam" << hnsw->upper_beam << std::endl;
    
    current_pos = get_reader_position(dynamic_cast<const faiss::VectorIOReader&>(*f));
    // std::cout << "read_dhnsw_HNSW done" << std::endl;
}
    static void read_dhnsw_HNSW_(
        HNSW* hnsw,
        IOReader* f,
        std::vector<uint64_t>& offset_para,
        int sub_idx,
        size_t& current_pos) {
    // std::cout << "read_dhnsw_HNSW_" << std::endl;
    size_t base_idx = sub_idx * 9;
    
    // Read levels
    current_pos = offset_para[base_idx + 1];
    seek_in_generic_reader(f, current_pos);
    READVECTOR(hnsw->levels);
    
    // Read offsets
    current_pos = offset_para[base_idx + 2];
    seek_in_generic_reader(f, current_pos);
    READVECTOR(hnsw->offsets);
    
    // Read neighbors
    current_pos = offset_para[base_idx + 3];
    seek_in_generic_reader(f, current_pos);
    READVECTOR(hnsw->neighbors);
    
    // Read remaining HNSW data
    current_pos = offset_para[base_idx + 4];
    seek_in_generic_reader(f, current_pos);
    READ1(hnsw->entry_point);
    // std::cout << "read entry_point" << hnsw->entry_point << std::endl;
    READ1(hnsw->max_level);
    // std::cout << "read max_level" << hnsw->max_level << std::endl;
    READ1(hnsw->efConstruction);
    // std::cout << "read efConstruction" << hnsw->efConstruction << std::endl;
    READ1(hnsw->efSearch);
    // std::cout << "read efSearch" << hnsw->efSearch << std::endl;
    READ1(hnsw->upper_beam);
    // std::cout << "read upper_beam" << hnsw->upper_beam << std::endl;
    
    current_pos = get_generic_reader_position(f);
    // std::cout << "read_dhnsw_HNSW done" << std::endl;
    }
    
    static void read_dhnsw_storage(
        IndexFlat* storage,
        IOReader* f,
        std::vector<uint64_t>& offset_para,
        int sub_idx,
        size_t& current_pos) {
        
        // std::cout << "Starting read_dhnsw_storage at position: " << current_pos << std::endl;
        
        uint32_t h;
        READ1(h);
        current_pos = get_generic_reader_position(f); 

        // Ensure storage is properly initialized
        if (!storage) {
            std::cerr << "Error: Null storage pointer provided to read_dhnsw_storage" << std::endl;
            throw std::runtime_error("Null storage pointer in read_dhnsw_storage");
        }
        // std::cout << "Using existing IndexFlat storage at " << storage << std::endl;
  
        // Read storage data
        // current_pos = offset_para[sub_idx * 15 + 7] - sizeof(storage->d); // TODO: check if this is correct
        // seek_in_reader(dynamic_cast<faiss::VectorIOReader&>(*f), current_pos);
        read_original_index_header(storage, f);
        
        current_pos = offset_para[sub_idx * 15 + 7] + sizeof(storage->ntotal) + 2 * sizeof(Index::idx_t) + 
                    sizeof(storage->is_trained) + sizeof(storage->metric_type);
        if (storage->metric_type > 1) {
            current_pos += sizeof(storage->metric_arg);
        }
        current_pos = offset_para[sub_idx * 15 + 8];
        READVECTOR(storage->xb);
        current_pos = get_generic_reader_position(f);
        // the current_pos isn't updated, do not need ?
        
        
        // std::cout << "Finished read_dhnsw_storage at position: " << current_pos << std::endl;
    }

    static void read_dhnsw_storage_(
        IndexFlat* storage,
        IOReader* f,
        std::vector<uint64_t>& offset_para,
        int sub_idx,
        size_t& current_pos) {
        
        // std::cout << "Starting read_dhnsw_storage at position: " << current_pos << std::endl;
        
        uint32_t h;
        READ1(h);
        current_pos = get_generic_reader_position(f); 

        // Ensure storage is properly initialized
        if (!storage) {
            std::cerr << "Error: Null storage pointer provided to read_dhnsw_storage" << std::endl;
            throw std::runtime_error("Null storage pointer in read_dhnsw_storage");
        }
        // std::cout << "Using existing IndexFlat storage at " << storage << std::endl;
  
        // Read storage data
        // current_pos = offset_para[sub_idx * 15 + 7] - sizeof(storage->d); // TODO: check if this is correct
        // seek_in_reader(dynamic_cast<faiss::VectorIOReader&>(*f), current_pos);
        read_original_index_header(storage, f);

        current_pos = offset_para[sub_idx * 9 + 7];
        READVECTOR(storage->xb);
        current_pos = get_generic_reader_position(f);
        // the current_pos isn't updated, do not need ?
        
        
        // std::cout << "Finished read_dhnsw_storage at position: " << current_pos << std::endl;
    }

    void read_dhnsw_index_init(
        Index* idx,
        IOReader* f,
        std::vector<uint64_t>& offset_para,
        int sub_idx,
        size_t& current_pos) {
        
        // std::cout << "Starting read_dhnsw_index_init for sub_idx: " << sub_idx << std::endl;
        
        // Check that idx is not null
        if (!idx) {
            std::cerr << "Error: Null index pointer provided to read_dhnsw_index_init" << std::endl;
            throw std::runtime_error("Null index pointer provided to read_dhnsw_index_init");
        }
        // std::cout << "Using existing IndexHNSWFlat at " << idx << std::endl;
         
        if (offset_para.size() < (sub_idx + 1) * 15) {
            throw std::runtime_error("Invalid offset parameters for sub HNSW deserialization");
        }
        
        seek_in_reader(dynamic_cast<faiss::VectorIOReader&>(*f), current_pos);
        uint32_t h;
        READ1(h);
        current_pos = get_reader_position(dynamic_cast<const faiss::VectorIOReader&>(*f)); 

        if (h != fourcc("IHNf")) {
            throw std::runtime_error("Invalid type in sub HNSW deserialization");
        }
        
        
        // std::cout << "Reading index header..." << std::endl;
         
        // std::cout << "get here" << std::endl; 
        read_dhnsw_index_header(idx, f, offset_para, sub_idx, current_pos);
        
        // std::cout << "Reading HNSW structure..." << std::endl;
        IndexHNSW* idxhnsw = dynamic_cast<IndexHNSW*>(idx);
        if (!idxhnsw) {
            throw std::runtime_error("Index is not HNSW type");
        }
        read_dhnsw_HNSW(&idxhnsw->hnsw, f, offset_para, sub_idx, current_pos);

        // std::cout << "Reading storage..." << std::endl;
        // Make sure storage is properly initialized for IndexHNSWFlat
        IndexHNSWFlat* idxhnswflat = dynamic_cast<IndexHNSWFlat*>(idx);
        if (idxhnswflat) {
            // Initialize storage if needed
            if (!idxhnswflat->storage) {
                idxhnswflat->storage = new IndexFlat(idxhnswflat->d);
                // std::cout << "Created new IndexFlat storage at " << idxhnswflat->storage << std::endl;
            }
            
            seek_in_reader(dynamic_cast<faiss::VectorIOReader&>(*f), current_pos);
            read_dhnsw_storage(dynamic_cast<IndexFlat*>(idxhnswflat->storage), f, offset_para, sub_idx, current_pos);
        } else {
            std::cerr << "Warning: Index is not IndexHNSWFlat, could not read storage" << std::endl;
        }
        
        // Verify the object was properly initialized
        // std::cout << "After initialization: max_level=" << idxhnsw->hnsw.max_level 
                //   << ", efSearch=" << idxhnsw->hnsw.efSearch
                //   << ", entry_point=" << idxhnsw->hnsw.entry_point << std::endl; 
    } 

    void read_dhnsw_index_init_(
        Index* idx,
        IOReader* f,
        std::vector<uint64_t>& offset_para,
        int sub_idx,
        size_t& current_pos) {
        
        // std::cout << "Starting read_dhnsw_index_init for sub_idx: " << sub_idx << std::endl;
        
        // Check that idx is not null
        if (!idx) {
            std::cerr << "Error: Null index pointer provided to read_dhnsw_index_init" << std::endl;
            throw std::runtime_error("Null index pointer provided to read_dhnsw_index_init");
        }
        // std::cout << "Using existing IndexHNSWFlat at " << idx << std::endl;
         
        if (offset_para.size() < (sub_idx + 1) * 9) {
            throw std::runtime_error("Invalid offset parameters for sub HNSW deserialization");
        }
        
        seek_in_generic_reader(f, current_pos);
        uint32_t h;
        READ1(h);
        current_pos = get_reader_position(dynamic_cast<const faiss::VectorIOReader&>(*f)); 

        if (h != fourcc("IHNf")) {
            throw std::runtime_error("Invalid type in sub HNSW deserialization");
        }
        
        
        // std::cout << "Reading index header..." << std::endl;
         
        // std::cout << "get here" << std::endl; 
        read_dhnsw_index_header(idx, f, offset_para, sub_idx, current_pos);
        
        // std::cout << "Reading HNSW structure..." << std::endl;
        IndexHNSW* idxhnsw = dynamic_cast<IndexHNSW*>(idx);
        if (!idxhnsw) {
            throw std::runtime_error("Index is not HNSW type");
        }
        read_dhnsw_HNSW_(&idxhnsw->hnsw, f, offset_para, sub_idx, current_pos);

        // std::cout << "Reading storage..." << std::endl;
        // Make sure storage is properly initialized for IndexHNSWFlat
        IndexHNSWFlat* idxhnswflat = dynamic_cast<IndexHNSWFlat*>(idx);
        if (idxhnswflat) {
            // Initialize storage if needed
            if (!idxhnswflat->storage) {
                idxhnswflat->storage = new IndexFlat(idxhnswflat->d);
                // std::cout << "Created new IndexFlat storage at " << idxhnswflat->storage << std::endl;
            }
            
            seek_in_reader(dynamic_cast<faiss::VectorIOReader&>(*f), current_pos);
            read_dhnsw_storage_(dynamic_cast<IndexFlat*>(idxhnswflat->storage), f, offset_para, sub_idx, current_pos);
        } else {
            std::cerr << "Warning: Index is not IndexHNSWFlat, could not read storage" << std::endl;
        }
        
        // Verify the object was properly initialized
        // std::cout << "After initialization: max_level=" << idxhnsw->hnsw.max_level 
                //        << ", efSearch=" << idxhnsw->hnsw.efSearch
                //   << ", entry_point=" << idxhnsw->hnsw.entry_point << std::endl;
    }  

    size_t get_reader_position(const faiss::VectorIOReader& reader) {
        return reader.rp;
    }

    void read_dhnsw_single_sub_hnsw(
        IndexHNSW* idx,
        const std::vector<uint8_t>& data,
        std::vector<uint64_t>& offset_para,
        std::vector<uint64_t>& offset_sub_hnsw,
        int sub_idx) {
        
        // Initialize reader with the extracted data
        VectorIOReader f;
        f.data = data;
        f.rp = 0;
        size_t base_idx = sub_idx * 15;

        // Verify index pointer
        if (!idx) {
            idx = new IndexHNSWFlat();
            if (!idx) {
                throw std::runtime_error("Failed to allocate new IndexHNSWFlat");
            }
        }
        for (int i = 0; i < 15; i++) {
            offset_para[base_idx + i] -= offset_sub_hnsw[sub_idx * 2];
        }
        size_t current_pos = 0;
        read_dhnsw_index_init(idx, &f, offset_para, sub_idx, current_pos);
        for (int i = 0; i < 15; i++) {
            offset_para[base_idx + i] += offset_sub_hnsw[sub_idx * 2];
        }
}

void read_HNSW_optimized(faiss::HNSW* hnsw, faiss::IOReader* f) {
    uint64_t assign_probas_size = 0;
    READ1(assign_probas_size);
    hnsw->assign_probas.resize(assign_probas_size);
    fast_read_exact(f, hnsw->assign_probas.data(), sizeof(double), assign_probas_size);
    uint64_t cum_nneighbor_per_level_size = 0;
    READ1(cum_nneighbor_per_level_size);
    hnsw->cum_nneighbor_per_level.resize(cum_nneighbor_per_level_size);
    fast_read_exact(f, hnsw->cum_nneighbor_per_level.data(), sizeof(int), cum_nneighbor_per_level_size);

   
    uint64_t levels_size = 0;
    READ1(levels_size);
    hnsw->levels.resize(levels_size);
 
    fast_read_exact(f, hnsw->levels.data(), sizeof(int), levels_size);
    uint64_t offsets_size = 0;
    READ1(offsets_size);
    hnsw->offsets.resize(offsets_size);
    fast_read_exact(f, hnsw->offsets.data(), sizeof(size_t), offsets_size);
    uint64_t neighbors_size = 0;
    READ1(neighbors_size);
    hnsw->neighbors.resize(neighbors_size);
    fast_read_exact(f, hnsw->neighbors.data(), sizeof(storage_idx_t), neighbors_size);
    READ1(hnsw->entry_point);

    // // std::cout << "read entry_point" << hnsw->entry_point << std::endl;
    READ1(hnsw->max_level);
    // // std::cout << "read max_level" << hnsw->max_level << std::endl;
    READ1(hnsw->efConstruction);
    // // std::cout << "read efConstruction" << hnsw->efConstruction << std::endl;
    READ1(hnsw->efSearch);
    // // std::cout << "read efSearch" << hnsw->efSearch << std::endl;
    READ1(hnsw->upper_beam);
    // // std::cout << "read upper_beam" << hnsw->upper_beam << std::endl;
}

void read_HNSW_optimized_(faiss::HNSW* hnsw, faiss::IOReader* f,
    std::vector<uint64_t>& offset_para, 
    std::vector<uint64_t>& overflow) {
    // uint64_t assign_probas_size = 0;
    // READ1(assign_probas_size);
    // // std::cout << "read assign_probas_size" << assign_probas_size << std::endl;
    // hnsw->assign_probas.resize(assign_probas_size);
    // fast_read_exact(f, hnsw->assign_probas.data(), sizeof(double), assign_probas_size);
    // uint64_t cum_nneighbor_per_level_size = 0;
    // READ1(cum_nneighbor_per_level_size);
    // // std::cout << "read cum_nneighbor_per_level_size" << cum_nneighbor_per_level_size << std::endl;
    // hnsw->cum_nneighbor_per_level.resize(cum_nneighbor_per_level_size);
    // fast_read_exact(f, hnsw->cum_nneighbor_per_level.data(), sizeof(int), cum_nneighbor_per_level_size);

    seek_in_generic_reader(f, offset_para[1]); 
    uint64_t levels_size = 0;
    READ1(levels_size);
    // std::cout << "read levels_size" << levels_size << std::endl;
    hnsw->levels.resize(levels_size);
    fast_read_exact(f, hnsw->levels.data(), sizeof(int), levels_size);
    // std::cout << "read levels" << std::endl;
    seek_in_generic_reader(f, offset_para[2]);
    uint64_t offsets_size = 0;
    READ1(offsets_size);
    // std::cout << "read offsets_size" << offsets_size << std::endl;
    hnsw->offsets.resize(offsets_size);
    fast_read_exact(f, hnsw->offsets.data(), sizeof(size_t), offsets_size);
    // std::cout << "read offsets" << std::endl;
    seek_in_generic_reader(f, offset_para[3]);
    uint64_t neighbors_size = 0;
    READ1(neighbors_size);
    // std::cout << "read neighbors_size" << neighbors_size << std::endl;
    hnsw->neighbors.resize(neighbors_size);
    fast_read_exact(f, hnsw->neighbors.data(), sizeof(storage_idx_t), neighbors_size);
    // std::cout << "read neighbors end" << std::endl;
    seek_in_generic_reader(f, offset_para[4]);
    READ1(hnsw->entry_point);

    // std::cout << "read entry_point" << hnsw->entry_point << std::endl;
    READ1(hnsw->max_level);
    // std::cout << "read max_level" << hnsw->max_level << std::endl;
    READ1(hnsw->efConstruction);
    // std::cout << "read efConstruction" << hnsw->efConstruction << std::endl;
    READ1(hnsw->efSearch);
    // std::cout << "read efSearch" << hnsw->efSearch << std::endl;
    READ1(hnsw->upper_beam);
    // std::cout << "read upper_beam" << hnsw->upper_beam << std::endl;
}

faiss::IndexHNSWFlat* read_index_HNSWFlat_optimized(faiss::IOReader* f) {
    uint32_t h;
    READ1(h);
    if (h != fourcc("IHNf")) {
        throw std::runtime_error("Invalid sub-HNSW type in optimized read");
    }
    // // std::cout << "read h" << h << std::endl;
    faiss::IndexHNSWFlat* idx = new faiss::IndexHNSWFlat();
    {
        READ1(idx->d);
        // // std::cout << "read d" << idx->d << std::endl;
        READ1(idx->ntotal);
        // // std::cout << "read ntotal" << idx->ntotal << std::endl;
        faiss::Index::idx_t dummy;
        READ1(dummy);
        // // std::cout << "read dummy" << dummy << std::endl;
        READ1(dummy);
        // // std::cout << "read dummy" << dummy << std::endl;
        READ1(idx->is_trained);
        // // std::cout << "read is_trained" << idx->is_trained << std::endl;
        READ1(idx->metric_type);
        // // std::cout << "read metric_type" << idx->metric_type << std::endl;
        if (idx->metric_type > 1) {
            READ1(idx->metric_arg);
            // // std::cout << "read metric_arg" << idx->metric_arg << std::endl;
        }
        // // std::cout << "read metric_arg done" << std::endl;
        idx->verbose = false;
    }

    read_HNSW_optimized(&idx->hnsw, f);


    idx->storage = read_storage_optimized(f);
    // // std::cout << "read storage done" << std::endl;  
    idx->own_fields = true;

    return idx;
}

faiss::IndexHNSWFlat* read_index_HNSWFlat_optimized_(faiss::IOReader* f, 
    std::vector<uint64_t>& offset_para, 
    std::vector<uint64_t>& overflow) {
        int32_t h;
    READ1(h);
    if (h != fourcc("IHNf")) {
        throw std::runtime_error("Invalid sub-HNSW type in optimized read");
    }
    // // std::cout << "read h" << h << std::endl;
    faiss::IndexHNSWFlat* idx = new faiss::IndexHNSWFlat();
    {
        READ1(idx->d);
        // std::cout << "read d" << idx->d << std::endl;
        READ1(idx->ntotal);
        // std::cout << "read ntotal" << idx->ntotal << std::endl;
        faiss::Index::idx_t dummy;
        READ1(dummy);
        // std::cout << "read dummy" << dummy << std::endl;
        READ1(dummy);
        // std::cout << "read dummy" << dummy << std::endl;
        READ1(idx->is_trained);
        // std::cout << "read is_trained" << idx->is_trained << std::endl;
        READ1(idx->metric_type);
        // std::cout << "read metric_type" << idx->metric_type << std::endl;
        if (idx->metric_type > 1) {
            READ1(idx->metric_arg);
            // std::cout << "read metric_arg" << idx->metric_arg << std::endl;
        }
        // std::cout << "read metric_arg done" << std::endl;
        idx->verbose = false;
    }

    read_HNSW_optimized_(&idx->hnsw, f, offset_para, overflow);


    idx->storage = read_storage_optimized_(f, offset_para, overflow);
    // std::cout << "read storage done" << std::endl;  
    idx->own_fields = true;

    return idx;
}
faiss::Index* read_storage_optimized(faiss::IOReader* f) {
        uint32_t h;
        READ1 (h);
        if (h != fourcc("IxF2")) {
            throw std::runtime_error("Invalid storage type in optimized read");
        }
        faiss::IndexFlatL2* idxf = new faiss::IndexFlatL2 ();
        READ1 (idxf->d);
        READ1 (idxf->ntotal);
        faiss::Index::idx_t dummy;
        READ1 (dummy);
        READ1 (dummy);
        READ1 (idxf->is_trained);
        READ1 (idxf->metric_type);
        if (idxf->metric_type > 1) {
            READ1 (idxf->metric_arg);
        }
        idxf->verbose = false;
        uint64_t xb_size = 0;
        READ1(xb_size); 
        idxf->xb.resize(xb_size);
  
        fast_read_exact(f, idxf->xb.data(), sizeof(float), xb_size);
        FAISS_THROW_IF_NOT (idxf->xb.size() == idxf->ntotal * idxf->d);
        return idxf;
    }

inline void fast_read_exact(IOReader* f, void* dst, size_t item_size, size_t item_count) {
    size_t got = f->operator()(dst, item_size, item_count);
    if (got != item_count) {
        throw std::runtime_error("IO read error");
    }
}
faiss::Index* read_storage_optimized_(faiss::IOReader* f,
    std::vector<uint64_t>& offset_para, 
    std::vector<uint64_t>& overflow) {
        uint32_t h;
        READ1 (h);
        if (h != fourcc("IxF2")) {
            throw std::runtime_error("Invalid storage type in optimized read");
        }
        faiss::IndexFlatL2* idxf = new faiss::IndexFlatL2 ();
        READ1 (idxf->d);
        READ1 (idxf->ntotal);
        faiss::Index::idx_t dummy;
        READ1 (dummy);
        READ1 (dummy);
        READ1 (idxf->is_trained);
        READ1 (idxf->metric_type);
        if (idxf->metric_type > 1) {
            READ1 (idxf->metric_arg);
        }
        idxf->verbose = false;
         
        seek_in_generic_reader(f, offset_para[7]);
        uint64_t xb_size = 0;
        READ1(xb_size); 
        idxf->xb.resize(xb_size);
        // std::cout << "read storage done" << std::endl; 
        fast_read_exact(f, idxf->xb.data(), sizeof(float), xb_size);
        FAISS_THROW_IF_NOT (idxf->xb.size() == idxf->ntotal * idxf->d);
        return idxf;
    }
    void seek_in_reader(faiss::VectorIOReader& reader, size_t pos) {
        if (pos <= reader.data.size()) {
            reader.rp = pos;
        } else {
            throw std::out_of_range("Seek position is out of range for VectorIOReader");
        }
    }
    void seek_in_reader(DirectMemoryIOReader& reader, size_t pos) {
        if (pos <= reader.total_size) {
            reader.current_ptr = reader.start_ptr + pos;
        } else {
            throw std::out_of_range("Seek position is out of range for DirectMemoryIOReader");
        }
    }
    void seek_in_generic_reader(IOReader* f, size_t pos) {
        if (auto* reader_vec = dynamic_cast<faiss::VectorIOReader*>(f)) {
            seek_in_reader(*reader_vec, pos);
        } else if (auto* reader_direct = dynamic_cast<DirectMemoryIOReader*>(f)) {
            seek_in_reader(*reader_direct, pos);
        } else {
            throw std::runtime_error("Unsupported IOReader type for seeking");
        }
    }
    size_t get_generic_reader_position(IOReader* f) {
        if (auto* reader_vec = dynamic_cast<faiss::VectorIOReader*>(f)) {
            return reader_vec->rp;
        } else if (auto* reader_direct = dynamic_cast<DirectMemoryIOReader*>(f)) {
            return reader_direct->tell();
        }
        throw std::runtime_error("Unsupported IOReader type for tell");
    }
}


