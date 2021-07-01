#include "squeeze.h"

namespace metawarenn {

namespace op {

Squeeze::Squeeze(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Squeeze") {
  inputs = n_inputs;
  outputs = n_outputs;
  }
} //namespace op

} //namespace metawarenn
