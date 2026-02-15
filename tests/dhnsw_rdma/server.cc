#include <gflags/gflags.h>

#include "../../deps/rlibv2/lib.hh"
#include <iostream>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include "../dhnsw/DistributedHnsw.h"
#include <iostream>
#include <fstream>
#include "../util/read_dataset.h"
#include <chrono>
DEFINE_int64(port, 8888, "Server listener (UDP) port.");
DEFINE_int64(use_nic_idx, 3, "Which NIC to create QP");
DEFINE_int64(reg_nic_name, 73, "The name to register an opened NIC at rctrl.");
DEFINE_int64(reg_mem_name, 73, "The name to register an MR at rctrl.");

using namespace rdmaio;
using namespace rdmaio::rmem;

int main(int argc, char **argv) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // start a controler, so that others may access it using UDP based channel
  RCtrl ctrl(FLAGS_port);
  RDMA_LOG(4) << "Pingping server listenes at localhost:" << FLAGS_port;

  // open the NIC
  auto nic =
      RNic::create(RNicInfo::query_dev_names().at(FLAGS_use_nic_idx)).value();

  // register the nic with name 0 to the ctrl
  RDMA_ASSERT(ctrl.opened_nics.reg(FLAGS_reg_nic_name, nic));

  size_t fixed_size = 1024*1024*20;
  RDMA_LOG(4) << "Allocating fixed-size buffer of size: " << fixed_size;
  
  //initialize the server side dhnsw
  int dim = 128; 
  int num_data = 10000;
  int num_meta = 50;
  int num_sub_hnsw = 20;
  int meta_hnsw_neighbors = 8;
  int sub_hnsw_neighbors = 32;
  std::string base_data_path = "../datasets/siftsmall/siftsmall_base.fvecs";
  std::vector<float> base_data;
  int dim_base_data;
  int n_base_data;
  base_data = read_fvecs(base_data_path, dim_base_data, n_base_data);
  std::cout << dim_base_data << std::endl;
  std::cout << n_base_data << std::endl;
  std::cout << "Read data successfully!" << std::endl;
  auto start_build = std::chrono::high_resolution_clock::now();
  DistributedHnsw dhnsw(dim, num_sub_hnsw, meta_hnsw_neighbors, sub_hnsw_neighbors, num_meta);
  dhnsw.build(base_data);
  // copy the dhnsw to the registered memory
 
  // Ensure the memory region is large enough to hold the DistributedHnsw object
  std::vector<uint8_t> serialized_data = dhnsw.serialize();
  size_t data_size = serialized_data.size();
  RDMA_LOG(4) << "Serialized data size: " << data_size;

  auto mr_memory = Arc<RMem>(new RMem(data_size + sizeof(size_t)));
  RDMA_ASSERT(ctrl.registered_mrs.create_then_reg(
        FLAGS_reg_mem_name, mr_memory, nic));
  if (data_size + sizeof(size_t) > fixed_size) {
        RDMA_LOG(4) << "Serialized data size exceeds fixed buffer size";
        return -1;
  }

  
  auto mr_attr = ctrl.registered_mrs.query(FLAGS_reg_mem_name).value()->get_reg_attr().value();
  uint8_t* reg_mem = reinterpret_cast<uint8_t*>(mr_attr.buf);
  // Copy the data size  first into the buffer
  std::memcpy(reg_mem, &data_size, sizeof(size_t));
  // Copy serialized data into the registered memory after the size
  std::memcpy(reg_mem + sizeof(size_t), serialized_data.data(), data_size);

  // start the listener thread so that client can communicate w it
  ctrl.start_daemon();
  std::cout << "remote address" << mr_attr.key << std::endl;
  RDMA_LOG(2) << "RC pingpong server started!";
  // run for 20 seconds
 while (true) {
    // Accept connections and handle them
    // Ensure that QPs are properly set up and remain active
    sleep(1); // Keep the server alive
  } 
  RDMA_LOG(4) << "server exit!";
}