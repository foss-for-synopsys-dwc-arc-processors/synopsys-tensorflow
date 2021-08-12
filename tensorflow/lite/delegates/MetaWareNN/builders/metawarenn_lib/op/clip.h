#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class Clip : public Node {
  public:
    Clip();
    Clip(std::string n_name, std::vector<std::string> n_inputs,
        std::vector<std::string> n_outputs,
        std::vector<float> n_min, std::vector<float> n_max);
    void fill_attributes(DataSerialization &layer_serializer) override;
  private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<float> min;
    std::vector<float> max;
};

} //namespace op

} //namespace metawarenn
