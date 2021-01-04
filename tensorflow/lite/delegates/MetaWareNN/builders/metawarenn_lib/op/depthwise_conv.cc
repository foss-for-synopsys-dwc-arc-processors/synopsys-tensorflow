#include "depthwise_conv.h"

namespace metawarenn {

namespace op {

DepthwiseConv::DepthwiseConv(std::string n_name, std::vector<std::string> n_inputs,
                             std::vector<std::string> n_outputs,
                             std::vector<int> n_dilations,
                             std::vector<int> n_strides,
                             std::vector<int> n_pads) : Node(n_name, "DepthwiseConv") {
  inputs = n_inputs;
  outputs = n_outputs;
  dilations = n_dilations;
  strides = n_strides;
  pads = n_pads;
  }
} //namespace op

} //namespace metawarenn
