#include "transpose.h"

namespace metawarenn {

namespace op {

Transpose::Transpose(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs) : Node(n_name, "Transpose") {
  inputs = n_inputs;
  outputs = n_outputs;
  }
} //namespace op

} //namespace metawarenn
