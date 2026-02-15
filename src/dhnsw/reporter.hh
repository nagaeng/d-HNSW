#pragma once

#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include "./statics.hh"
#include "../../deps/r2/src/logging.hh"

namespace bench {

template<class T>
std::string format_value(T value, int precission = 4) {
  std::stringstream ss;
  ss.imbue(std::locale(""));
  ss << std::fixed << std::setprecision(precission) << value;
  return ss.str();
}

class Reporter {
public:

  // Write header for batch CSV file
  static void write_header(std::ofstream& out) {
    out << "batch_id,compute_time_us,network_latency_us,deserialize_time_us\n";
  }

  // Report per-batch metrics 
  static void report_batch(Statics& stat, int batch_id, const std::string& log_file = "") {
    std::string msg = "[Batch " + std::to_string(batch_id) + "] "
                    + "Compute: " + format_value(stat.data.compute_time_us, 2) + " ms, "
                    + "Network: " + format_value(stat.data.network_latency_us, 2) + " ms, "
                    + "Deserialize: " + format_value(stat.data.deserialize_time_us, 2) + " ms, "
                    + "Meta Search: " + format_value(stat.data.meta_search_time_us, 2) + " ms, "
                    + "Throughput: " + format_value(stat.data.throughput_qps, 2) + " qps, "
                    + "Total Time: " + format_value(stat.data.total_time_us, 2) + " ms";
 

    // std::cout << msg << std::endl;

    // Use RDMA_LOG instead of r2::LOG
    LOG(2) << msg;

    // Write to log file if specified
    // if (!log_file.empty()) {
    //   std::ofstream outfile(log_file, std::ios::out | std::ios::app);
    //   if (outfile.is_open()) {
    //     outfile << msg << std::endl;
    //     outfile.close();
    //   } else {
    //     std::cerr << "Error: Could not open log file: " << log_file << std::endl;
    //   }
    // }
  }
};

} // namespace bench