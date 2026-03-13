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
DEFINE_string(addr, "192.168.1.2:8888", "Server address to connect to.");
DEFINE_int64(use_nic_idx, 3, "Which NIC to create QP");
DEFINE_int64(reg_nic_name, 73, "The name to register an opened NIC at rctrl.");
DEFINE_int64(reg_mem_name, 73, "The name to register an MR at rctrl.");

using namespace rdmaio;
using namespace rdmaio::rmem;
using namespace rdmaio::qp;

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // 1. create a local QP to use
  auto nic =
      RNic::create(RNicInfo::query_dev_names().at(FLAGS_use_nic_idx)).value();
  auto qp = RC::create(nic, QPConfig()).value();

  // 2. create the pair QP at server using CM
  ConnectManager cm(FLAGS_addr);
  if (cm.wait_ready(1000000, 20) ==
      IOCode::Timeout) // wait 1 second for server to ready, retry 2 times
    RDMA_ASSERT(false) << "cm connect to server timeout";

  auto qp_res = cm.cc_rc("client-qp", qp, FLAGS_reg_nic_name, QPConfig());
  RDMA_ASSERT(qp_res == IOCode::Ok) << std::get<0>(qp_res.desc);
  auto key = std::get<1>(qp_res.desc);
  RDMA_LOG(4) << "client fetch QP authentical key: " << key;

  // 3. create the local MR for usage, and create the remote MR for usage
  size_t fixed_size =  1024*1024*1024;
  auto local_mem = Arc<RMem>(new RMem(fixed_size));
  MemoryFlags flags;
  flags.set_flags(IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                  IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC);
  auto local_mr = RegHandler::create(local_mem, nic, flags).value();

  auto fetch_res = cm.fetch_remote_mr(FLAGS_reg_mem_name);
  RDMA_ASSERT(fetch_res == IOCode::Ok) << std::get<0>(fetch_res.desc);
  rmem::RegAttr remote_attr = std::get<1>(fetch_res.desc);
  std::cout << "remote address" << remote_attr.key << std::endl;
  qp->bind_remote_mr(remote_attr);
  qp->bind_local_mr(local_mr->get_reg_attr().value());
  // Allocate buffer to hold the received data
  uint8_t* recv_buffer = reinterpret_cast<uint8_t*>(local_mem->raw_ptr);
  std::cout << "Fetched remote MR attributes:" << std::endl;
  std::cout << "Remote address: " << remote_attr.buf << std::endl;
  std::cout << "Remote key: " << remote_attr.key << std::endl;
  std::cout << "Remote size: " << remote_attr.sz << std::endl;

  // Read the data size first
  size_t size_to_read = sizeof(size_t);

  // Perform RDMA read to get the data size
  
  auto res_s = qp->send_normal(
      rdmaio::qp::RC::ReqDesc{
          .op = IBV_WR_RDMA_READ,
          .flags = IBV_SEND_SIGNALED,
          .len = static_cast<uint32_t>(size_to_read),
          .wr_id = 0
      },
      rdmaio::qp::RC::ReqPayload{
          .local_addr = reinterpret_cast<RMem::raw_ptr_t>(recv_buffer),
          .remote_addr = 0,
          .imm_data = 0
      }
  );
  RDMA_ASSERT(res_s == IOCode::Ok);

    // Wait for completion
  auto res_p = qp->wait_one_comp();
  RDMA_ASSERT(res_p == IOCode::Ok);
  RDMA_LOG(4) << "RDMA read for data size completed";
  
    // Retrieve the data size
    size_t data_size = 0;
    std::memcpy(&data_size, recv_buffer, sizeof(size_t));
    RDMA_LOG(4) << "Data size to read: " << data_size;


ibv_context *context = nic->get_ctx();

// Query port attributes
struct ibv_port_attr port_attr;
int port_num = 1; // Adjust this to the correct port number for your device

if (ibv_query_port(context, port_num, &port_attr)) {
    RDMA_LOG(ERROR) << "Failed to query port attributes";
    return -1;
}

// Get the maximum RDMA read size
size_t max_rdma_read_size = port_attr.max_msg_sz;
RDMA_LOG(INFO) << "Maximum RDMA read size: " << max_rdma_read_size;
   // Read the actual data
    res_s = qp->send_normal(
    rdmaio::qp::RC::ReqDesc{
        .op = IBV_WR_RDMA_READ,
        .flags = IBV_SEND_SIGNALED,
        .len = static_cast<uint32_t>(data_size),
        .wr_id = 1
    },
    rdmaio::qp::RC::ReqPayload{
        .local_addr = reinterpret_cast<RMem::raw_ptr_t>(recv_buffer + sizeof(size_t)),
        .remote_addr = static_cast<uint64_t>(sizeof(size_t)),
        .imm_data = 0
    }
    );
    RDMA_ASSERT(res_s == IOCode::Ok);

    // Wait for completion
    res_p = qp->wait_one_comp();
    RDMA_ASSERT(res_p == IOCode::Ok);
    RDMA_LOG(4) << "RDMA read for data completed";

    // Deserialize the data
    std::vector<uint8_t> serialized_data(recv_buffer + sizeof(size_t), recv_buffer + sizeof(size_t) + data_size);
    DistributedHnsw dhnsw(0, 0, 0, 0, 0);
    dhnsw.deserialize(serialized_data);
    RDMA_LOG(4) << "Deserialization completed";
  /***********************************************************/
  // Test the hierarchical search
  int branching_k = 3;
  int top_k = 1;
  std::string query_data_path = "../datasets/siftsmall/siftsmall_query.fvecs";
  std::string ground_truth_path = "../datasets/siftsmall/siftsmall_groundtruth.ivecs";
  std::vector<float> query_data;
  std::vector<int> ground_truth;
  int dim_query_data;
  int n_query_data;
  int dim_ground_truth;
  int n_ground_truth;
  query_data = read_fvecs(query_data_path, dim_query_data, n_query_data);
  dhnsw_idx_t *labels = new dhnsw_idx_t[top_k * n_query_data];
  float *distances = new float[top_k * n_query_data];
  dhnsw_idx_t *sub_hnsw_tags = new dhnsw_idx_t[top_k * n_query_data];
  dhnsw_idx_t *original_index = new dhnsw_idx_t[top_k * n_query_data];

  dhnsw.hierarchicalSearch(n_query_data, query_data.data(), branching_k, top_k, distances, labels, sub_hnsw_tags, original_index); 
  int correct = 0;
  float recall = 0.0f;
  std::unordered_set<int> ground_truth_set;
  std::unordered_set<int> result_set;
  for(int i = 0; i < n_query_data; i++) {
    ground_truth_set.clear();
    result_set.clear();
    ground_truth_set.insert(ground_truth.begin() + i * 100, ground_truth.begin() + i * 100 + top_k);
    result_set.insert(original_index + i * top_k, original_index + (i + 1) * top_k);
    for(int j = 0; j < top_k; j++) {
      if(result_set.find(*ground_truth_set.begin()) != result_set.end()) {
        correct++;
      }
    }
  }
  recall = (float)correct / (n_query_data * top_k);
  RDMA_LOG(4) << "Recall: " << recall;
  // Clean up
    delete[] labels;
    delete[] distances;
    delete[] sub_hnsw_tags;
    delete[] original_index;

  /***********************************************************/

  // finally, some clean up, to delete my created QP at server
  auto del_res = cm.delete_remote_rc("client-qp", key);
  RDMA_ASSERT(del_res == IOCode::Ok)
      << "delete remote QP error: " << del_res.desc;

  RDMA_LOG(4) << "client returns";

  return 0;
}