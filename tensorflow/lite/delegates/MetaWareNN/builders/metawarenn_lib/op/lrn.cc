#include "lrn.h"

namespace metawarenn {

namespace op {

LRN::LRN(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs) : Node(n_name, "LRN") {
  inputs = n_inputs;
  outputs = n_outputs;
  }
} //namespace op

} //namespace metawarenn