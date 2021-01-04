#include "depthwise_conv.h"

namespace metawarenn {

namespace op {

DepthwiseConv::DepthwiseConv(std::string n_name, std::vector<std::string> n_inputs,
           std::vector<std::string> n_outputs) : Node(n_name, "DepthwiseConv") {
  inputs = n_inputs;
  outputs = n_outputs;
  }
} //namespace op

} //namespace metawarenn
