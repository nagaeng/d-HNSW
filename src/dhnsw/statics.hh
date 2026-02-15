#pragma once

#include "../../deps/r2/src/common.hh"

namespace bench {

using namespace r2;

/*!
 * Statistics used for multi-thread batch-level reporting.
 * The structure is 128-byte padded and aligned to avoid false sharing.
 */
struct alignas(128) Statics {

  typedef struct {
    // u64 counter = 0;
    // u64 counter1 = 0;
    // u64 counter2 = 0;
    // u64 counter3 = 0;
    // double lat = 0;
    // double lat1 = 0;

    // Per-batch metrics
    double compute_time_us = 0.0;
    double network_latency_us = 0.0;
    double deserialize_time_us = 0.0;
    double meta_search_time_us = 0.0;
    double throughput_qps = 0.0;
    double total_time_us = 0.0;
    int batch_id = -1;

  } data_t;

  data_t data;
  char pad[128 - sizeof(data)];

//   // Counter utils
//   void increment() { data.counter += 1; }
//   void increment_gap_1(u64 d) { data.counter1 += d; }
//   void set_lat(const double& l) { data.lat = l; }

  // Set all batch-level metrics
    void increment_compute(const double& l){
        data.compute_time_us += l;
    }
    void increment_network(const double& l){
        data.network_latency_us += l;
    }
    void increment_deserialize(const double& l){
        data.deserialize_time_us += l;
    }
    void increment_meta_search(const double& l){    
        data.meta_search_time_us += l;
    }
    void increment_throughput(const double& l){
        data.throughput_qps += l;
    }
    void set_batch_id(const int& l){
        data.batch_id = l;
    }
    void set_batch_metrics(const double& compute_time_us, const double& network_latency_us, const double& deserialize_time_us, const double& meta_search_time_us, const double& throughput_qps, const double& total_time_us, const int& batch_id){
        data.compute_time_us = compute_time_us;
        data.network_latency_us = network_latency_us;
        data.deserialize_time_us = deserialize_time_us;
        data.meta_search_time_us = meta_search_time_us;
        data.throughput_qps = throughput_qps;
        data.batch_id = batch_id;
        data.total_time_us = total_time_us;
    }
};

} // namespace bench