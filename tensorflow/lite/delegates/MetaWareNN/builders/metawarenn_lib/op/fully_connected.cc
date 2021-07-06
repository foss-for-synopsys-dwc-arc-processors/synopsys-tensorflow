#include "fully_connected.h"

namespace metawarenn {

namespace op {

FullyConnected::FullyConnected(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs) : Node(n_name, "FullyConnected") {
  inputs = n_inputs;
  outputs = n_outputs;
  }
} //namespace op

} //namespace metawarenn
