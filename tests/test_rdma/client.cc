#include <gflags/gflags.h>

#include "../../deps/rlibv2/core/lib.hh"
#include "../../deps/rlibv2/benchs/bench_op.hh"  // Include BenchOp template
#include <chrono>
DEFINE_string(addr, "192.168.1.2:8888", "Server address to connect to.");
DEFINE_int64(use_nic_idx, 3, "Which NIC to create QP");
DEFINE_int64(reg_nic_name, 73, "The name to register an opened NIC at rctrl.");
DEFINE_int64(reg_mem_name, 73, "The name to register an MR at rctrl.");

using namespace rdmaio;
using namespace rdmaio::rmem;
using namespace rdmaio::qp;
using namespace std::chrono;

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // 1. Create a local QP to use
  auto nic =
      RNic::create(RNicInfo::query_dev_names().at(FLAGS_use_nic_idx)).value();
  auto qp = RC::create(nic, QPConfig()).value();

  // 2. Create the pair QP at server using CM
  ConnectManager cm(FLAGS_addr);
  if (cm.wait_ready(1000000, 20) ==
      IOCode::Timeout)  // Wait 1 second for server to be ready, retry 2 times
    RDMA_ASSERT(false) << "cm connect to server timeout";

  auto qp_res = cm.cc_rc("client-qp", qp, FLAGS_reg_nic_name, QPConfig());
  RDMA_ASSERT(qp_res == IOCode::Ok) << std::get<0>(qp_res.desc);
  auto key = std::get<1>(qp_res.desc);
  RDMA_LOG(4) << "client fetch QP authentical key: " << key;

  // 3. Create the local MR for usage, and create the remote MR for usage
  const int total_ops = 12;
  auto local_mem = Arc<RMem>(new RMem(total_ops * sizeof(u64)));  // Allocate enough local memory
  auto local_mr = RegHandler::create(local_mem, nic).value();

  auto fetch_res = cm.fetch_remote_mr(FLAGS_reg_mem_name);
  RDMA_ASSERT(fetch_res == IOCode::Ok) << std::get<0>(fetch_res.desc);
  rmem::RegAttr remote_attr = std::get<1>(fetch_res.desc);

  qp->bind_remote_mr(remote_attr);
  qp->bind_local_mr(local_mr->get_reg_attr().value());

  /* This is the example code usage of the fully created RCQP */
  u64 *test_buf = (u64 *)(local_mem->raw_ptr);
  u64 *remote_buf = (u64 *)remote_attr.buf;

  // Initialize local buffer
  for (int i = 0; i < total_ops; ++i) {
    test_buf[i] = 0;
  }
  auto start1 = high_resolution_clock::now();
  const int batch_size = 4;  // Number of operations per batch
  BenchOp<> ops[batch_size];

  u32 lkey = local_mr->get_reg_attr().value().key;
  u32 rkey = remote_attr.key;

  int num_batches = total_ops / batch_size;
  int op_index = 0;

  for (int batch = 0; batch < num_batches; ++batch) {
    // For each operation in the batch, adjust the remote and local addresses
    for (int i = 0; i < batch_size; ++i) {
      ops[i].set_type(0);  // RDMA_READ
      u64 remote_offset = op_index * sizeof(u64);
      ops[i].init_rbuf((u64 *)((u64)remote_buf + remote_offset), rkey);
      ops[i].init_lbuf(test_buf + op_index, sizeof(u64), lkey);
      ops[i].set_wrid(op_index);
      ops[i].set_flags(0);  // Clear flags
      if (i != 0) {
        ops[i - 1].set_next(&ops[i]);  // Chain the operations
      }
      op_index++;
    }

    // Set the IBV_SEND_SIGNALED flag on the last operation to ring the doorbell
    ops[batch_size - 1].set_flags(IBV_SEND_SIGNALED);

    // Execute the batch
    auto res_s = ops[0].execute_batch(qp);
    RDMA_ASSERT(res_s == IOCode::Ok);

    // Wait for completion
    auto res_p = qp->wait_one_comp();
    RDMA_ASSERT(res_p == IOCode::Ok);

    std::cout << "doorbell ~" << std::endl;
    
    // Print the fetched values
    for (int i = op_index - batch_size; i < op_index; ++i) {
      
      RDMA_LOG(4) << "fetch value from server : 0x" << std::hex << test_buf[i];
    }
  }

  // Handle any remaining operations if total_ops is not a multiple of batch_size
  int remaining_ops = total_ops % batch_size;
  if (remaining_ops > 0) {
    // Adjust the batch size
    for (int i = 0; i < remaining_ops; ++i) {
      ops[i].set_type(0);  // RDMA_READ
      u64 remote_offset = op_index * sizeof(u64);
      ops[i].init_rbuf((u64 *)((u64)remote_buf + remote_offset), rkey);
      ops[i].init_lbuf(test_buf + op_index, sizeof(u64), lkey);
      ops[i].set_wrid(op_index);
      ops[i].set_flags(0);  // Clear flags
      if (i != 0) {
        ops[i - 1].set_next(&ops[i]);  // Chain the operations
      }
      op_index++;
    }

    ops[remaining_ops - 1].set_flags(IBV_SEND_SIGNALED);

    auto res_s = ops[0].execute_batch(qp);
    RDMA_ASSERT(res_s == IOCode::Ok);

    auto res_p = qp->wait_one_comp();
    RDMA_ASSERT(res_p == IOCode::Ok);

    for (int i = op_index - remaining_ops; i < op_index; ++i) {
      RDMA_LOG(4) << "fetch value from server : 0x" << std::hex << test_buf[i];
    }
  }
  auto stop1 = high_resolution_clock::now();
  auto duration = duration_cast<std::chrono::microseconds>(stop1 - start1);
  std::cout << "Time taken by doorbell: "
            << duration.count() << " microseconds" << std::endl;
  // without doorbell
  // Initialize local buffer
  for (int i = 0; i < total_ops; ++i) {
    test_buf[i] = 0;
  }
  auto start2 = high_resolution_clock::now();
  for (uint i = 0; i < 12; ++i) {
    auto res_s = qp->send_normal(
        {.op = IBV_WR_RDMA_READ,
         .flags = IBV_SEND_SIGNALED,
         .len = sizeof(u64),
         .wr_id = 0},
        {.local_addr = reinterpret_cast<RMem::raw_ptr_t>(test_buf),
         .remote_addr = i * sizeof(u64),
         .imm_data = 0});
    RDMA_ASSERT(res_s == IOCode::Ok);
    auto res_p = qp->wait_one_comp();
    RDMA_ASSERT(res_p == IOCode::Ok);

    RDMA_LOG(4) << "fetch one value from server : 0x" << std::hex <<  *test_buf;
  }
  auto stop2 = high_resolution_clock::now();
  auto duration2 = duration_cast<std::chrono::microseconds>(stop2 - start2);
  std::cout << "Time taken by normal: "
            << duration2.count() << " microseconds" << std::endl;
  // Clean up: delete the created QP at the server
  auto del_res = cm.delete_remote_rc("client-qp", key);
  RDMA_ASSERT(del_res == IOCode::Ok)
      << "delete remote QP error: " << del_res.desc;

  RDMA_LOG(4) << "client returns";

  return 0;
}