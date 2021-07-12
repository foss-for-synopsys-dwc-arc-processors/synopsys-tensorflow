#include "strided_slice.h"

namespace metawarenn {

namespace op {

StridedSlice::StridedSlice(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "StridedSlice") {
  inputs = n_inputs;
  outputs = n_outputs;
  }
} //namespace op

} //namespace metawarenn