#include "flatten.h"

namespace metawarenn {

namespace op {

Flatten::Flatten(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs) : Node(n_name, "Flatten") {
  inputs = n_inputs;
  outputs = n_outputs;
  }
} //namespace op

} //namespace metawarenn
