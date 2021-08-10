#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class FullyConnected : public Node {
  public:
    FullyConnected();
    FullyConnected(std::string n_name, std::vector<std::string> n_inputs,
            std::vector<std::string> n_outputs,
            std::vector<int> n_asymmetric_quantize_inputs,
            std::vector<int> n_keep_num_dims, std::vector<int> n_weights_format);
    void fill_attributes(DataSerialization &layer_serializer) override;
  private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<int> asymmetric_quantize_inputs;
    std::vector<int> keep_num_dims;
    std::vector<int> weights_format;
};

} //namespace op

} //namespace metawarenn
