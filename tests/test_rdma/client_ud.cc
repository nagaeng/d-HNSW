#include <assert.h>
#include <gflags/gflags.h>
#include <thread>

#include "../../deps/rlibv2/core/lib.hh"
#include "../../deps/rlibv2/core/qps/mod.hh"
#include "../../deps/rlibv2/core/utils/timer.hh"
#include "../../deps/rlibv2/core/qps/recv_iter.hh"

using namespace rdmaio;
using namespace rdmaio::qp;
using namespace rdmaio::rmem;

DEFINE_int64(port, 8888, "Server listener (UDP) port.");
DEFINE_int64(use_nic_idx, 3, "Which NIC to create QP");
DEFINE_int64(reg_mem_name, 73, "The name to register an MR at rctrl.");

class SimpleAllocator : public AbsRecvAllocator {
  RMem::raw_ptr_t buf = nullptr;
  usize total_mem = 0;
  mr_key_t key;

public:
  SimpleAllocator(Arc<RMem> mem, mr_key_t key)
      : buf(mem->raw_ptr), total_mem(mem->sz), key(key) {}

  Option<std::pair<rmem::RMem::raw_ptr_t, rmem::mr_key_t>>
  alloc_one(const usize &sz) override {
    if (total_mem < sz)
      return {};
    auto ret = buf;
    buf = static_cast<char *>(buf) + sz;
    total_mem -= sz;
    return std::make_pair(ret, key);
  }

  Option<std::pair<rmem::RMem::raw_ptr_t, rmem::RegAttr>>
  alloc_one_for_remote(const usize &sz) override {
    return {};
  }
};

// Function to deserialize QP attributes from a string
QPAttr deserialize_qp_attr(const std::string &data) {
  QPAttr attr;
  memcpy(&attr, data.data(), sizeof(QPAttr));
  return attr;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  RCtrl ctrl(FLAGS_port);
  RDMA_LOG(4) << "(UD) Server listens at localhost:" << FLAGS_port;

  // Open the NIC
  auto nic =
      RNic::create(RNicInfo::query_dev_names().at(FLAGS_use_nic_idx)).value();

  // Create UD QP with QKey 73
  auto ud_server = UD::create(nic, QPConfig().set_qkey(73)).value();
  ctrl.registered_qps.reg("server_ud", ud_server);

  // Prepare receive memory and allocator
  auto mem = Arc<RMem>(new RMem(16 * 1024 * 1024));
  auto handler = RegHandler::create(mem, nic).value();
  SimpleAllocator alloc(mem, handler->get_reg_attr().value().key);

  // Prepare receive entries
  auto recv_rs = RecvEntriesFactory<SimpleAllocator, 1024, 4096>::create(alloc);
  ctrl.registered_mrs.reg(FLAGS_reg_mem_name, handler);

  // Post receive buffers to the UD QP
  {
    recv_rs->sanity_check();
    auto res = ud_server->post_recvs(*recv_rs, 1024);
    RDMA_ASSERT(res == IOCode::Ok);
  }

  // Start the control daemon
  ctrl.start_daemon();

  RDMA_LOG(2) << "Server is ready to receive messages.";

  // Variables to store the client's QP attributes and AH
  QPAttr client_qp_attr;
  ibv_ah *client_ah = nullptr;

  // Receive loop
  constexpr size_t kGRHSz = 40;
  while (true) {
    for (RecvIter<UD, 1024> iter(ud_server, recv_rs); iter.has_msgs(); iter.next()) {
      auto imm_msg_opt = iter.cur_msg();
      if (!imm_msg_opt.has_value()) {
        RDMA_LOG(4) << "Received an invalid message.";
        continue;
      }
      auto imm_msg = imm_msg_opt.value();
      auto wc = iter.cur_wc();

      // Calculate the message length
      size_t msg_len = wc.wc.byte_len - kGRHSz;
      if (msg_len <= 0) {
        RDMA_LOG(2) << "Received empty message.";
        continue;
      }

      // Extract the message
      auto buf_ptr = static_cast<char *>(std::get<1>(imm_msg)) + kGRHSz;
      std::string msg(buf_ptr, msg_len);

      // Check if we have received the client's QP attributes
      if (msg_len == sizeof(QPAttr) && client_ah == nullptr) {
        // Deserialize the client's QP attributes
        client_qp_attr = deserialize_qp_attr(msg);
        // Create an AH for the client
        ibv_ah_attr ah_attr = {};
        ah_attr.is_global = 0;
        ah_attr.dlid = wc.wc.slid;
        ah_attr.sl = wc.wc.sl;
        ah_attr.src_path_bits = 0;
        ah_attr.port_num = nic->port_attr().port_id;

        client_ah = ibv_create_ah(nic->pd, &ah_attr);
        RDMA_ASSERT(client_ah != nullptr) << "Failed to create AH for client.";

        RDMA_LOG(2) << "Received client's QP attributes. Bidirectional communication established.";
        continue;
      }

      RDMA_LOG(2) << "Server received a message: " << msg;

      // If client AH is available, send a response
      if (client_ah != nullptr) {
        // Prepare the send WR
        ibv_send_wr wr = {};
        ibv_sge sge = {};

        wr.opcode = IBV_WR_SEND_WITH_IMM;
        wr.num_sge = 1;
        wr.imm_data = 73;
        wr.sg_list = &sge;

        wr.wr.ud.ah = client_ah;
        wr.wr.ud.remote_qpn = client_qp_attr.qpn;
        wr.wr.ud.remote_qkey = client_qp_attr.qkey;
        wr.send_flags = IBV_SEND_SIGNALED;

        // Prepare a response message
        std::string response = "Echo from server: " + msg;
        char *send_buf = buf_ptr; // Reuse the receive buffer for sending
        memcpy(send_buf, response.data(), response.size());

        sge.addr = reinterpret_cast<uintptr_t>(send_buf);
        sge.length = response.size();
        sge.lkey = handler->get_reg_attr().value().key;

        struct ibv_send_wr *bad_sr = nullptr;
        int ret = ibv_post_send(ud_server->qp, &wr, &bad_sr);
        if (ret) {
          RDMA_LOG(4) << "Failed to post send: " << strerror(ret);
          continue;
        }

        // Wait for completion
        auto ret_r = ud_server->wait_one_comp();
        if (ret_r != IOCode::Ok) {
          RDMA_LOG(4) << "Failed to get completion: " << UD::wc_status(ret_r.desc);
          continue;
        }

        RDMA_LOG(2) << "Server sent a response.";
      }
    }
  }

  return 0;

}
// ./server --port=8888 --use_nic_idx=3
// ./client --addr="192.168.1.2:8888" --use_nic_idx=3