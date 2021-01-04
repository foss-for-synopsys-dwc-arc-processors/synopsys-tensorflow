#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class DepthwiseConv : public Node {
  public:
    DepthwiseConv(std::string n_name, std::vector<std::string> n_inputs,
                  std::vector<std::string> n_outputs,
                  std::vector<int> n_dilations,
                  std::vector<int> n_strides,
                  std::vector<int> n_pads);
  private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<int> dilations;
    std::vector<int> strides;
    std::vector<int> pads;
};

} //namespace op

} //namespace metawarenn
