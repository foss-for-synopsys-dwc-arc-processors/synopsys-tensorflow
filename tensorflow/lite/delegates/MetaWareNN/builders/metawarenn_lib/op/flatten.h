#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class Flatten : public Node {
  public:
    Flatten();
    Flatten(std::string n_name, std::vector<std::string> n_inputs,
            std::vector<std::string> n_outputs, std::vector<int> n_axis);
    void fill_attributes(DataSerialization &layer_serializer) override;
  private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<int> axis;
};

} //namespace op

} //namespace metawarenn
