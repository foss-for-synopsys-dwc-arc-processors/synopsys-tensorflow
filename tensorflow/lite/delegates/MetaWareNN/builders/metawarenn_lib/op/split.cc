#include "split.h"

namespace metawarenn {

namespace op {

Split::Split(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Split") {
  inputs = n_inputs;
  outputs = n_outputs;
  }
} //namespace op

} //namespace metawarenn
