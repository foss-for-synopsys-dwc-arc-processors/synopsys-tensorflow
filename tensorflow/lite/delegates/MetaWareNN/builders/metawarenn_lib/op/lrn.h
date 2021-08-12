#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class LRN : public Node {
  public:
    LRN();
    LRN(std::string n_name, std::vector<std::string> n_inputs,
                      std::vector<std::string> n_outputs,
                      std::vector<int> n_alpha, std::vector<int> n_beta,
                      std::vector<int> n_axis, std::vector<int> n_size,
                      std::vector<int> n_bias, std::vector<int> n_half_window_size);
    void fill_attributes(DataSerialization &layer_serializer) override;
  private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<int> alpha;
    std::vector<int> beta;
    std::vector<int> axis;
    std::vector<int> size;
    std::vector<int> bias;
    std::vector<int> half_window_size;
};

} //namespace op

} //namespace metawarenn
