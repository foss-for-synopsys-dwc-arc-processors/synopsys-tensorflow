#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class MaxPool : public Node {
  public:
    MaxPool();
    MaxPool(std::string n_name, std::vector<std::string> n_inputs,
            std::vector<std::string> n_outputs,
                std::vector<int> n_kernel_shape,
                std::vector<int> n_strides,
                std::vector<int> n_pads);
    void fill_attributes(DataSerialization &layer_serializer) override;
  private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<int> kernel_shape;
    std::vector<int> strides;
    std::vector<int> pads;
};

} //namespace op

} //namespace metawarenn
