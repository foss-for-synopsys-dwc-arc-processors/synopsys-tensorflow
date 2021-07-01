#include "unsqueeze.h"

namespace metawarenn {

namespace op {

Unsqueeze::Unsqueeze(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Unsqueeze") {
  inputs = n_inputs;
  outputs = n_outputs;
  }
} //namespace op

} //namespace metawarenn
