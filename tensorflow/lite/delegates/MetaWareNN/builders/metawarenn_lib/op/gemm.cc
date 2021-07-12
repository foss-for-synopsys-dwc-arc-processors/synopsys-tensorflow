#include "gemm.h"

namespace metawarenn {

namespace op {

Gemm::Gemm(std::string n_name, std::vector<std::string> n_inputs,
           std::vector<std::string> n_outputs) : Node(n_name, "Gemm") {
  inputs = n_inputs;
  outputs = n_outputs;
  }
} //namespace op

} //namespace metawarenn
