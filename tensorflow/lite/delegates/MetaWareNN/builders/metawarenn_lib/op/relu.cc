#include "relu.h"

namespace metawarenn {

namespace op {

Relu::Relu(std::string n_name, std::vector<std::string> n_inputs,
           std::vector<std::string> n_outputs) : Node(n_name, "Relu") {
  inputs = n_inputs;
  outputs = n_outputs;
  }
} //namespace op

} //namespace metawarenn
