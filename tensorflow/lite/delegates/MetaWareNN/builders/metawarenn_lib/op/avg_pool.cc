#include "avg_pool.h"

namespace metawarenn {

namespace op {

AvgPool::AvgPool(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs) : Node(n_name, "AvgPool") {
  inputs = n_inputs;
  outputs = n_outputs;
  }
} //namespace op

} //namespace metawarenn
