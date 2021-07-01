#include "max_pool.h"

namespace metawarenn {

namespace op {

MaxPool::MaxPool(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs) : Node(n_name, "MaxPool") {
  inputs = n_inputs;
  outputs = n_outputs;
  }
} //namespace op

} //namespace metawarenn
