#include "conv.h"

namespace metawarenn {

namespace op {

Conv::Conv(std::string n_name, std::vector<std::string> n_inputs,
           std::vector<std::string> n_outputs) : Node(n_name, "Conv") {
  inputs = n_inputs;
  outputs = n_outputs;
  }
} //namespace op

} //namespace metawarenn
