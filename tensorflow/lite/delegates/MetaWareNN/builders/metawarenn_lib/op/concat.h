#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class Concat : public Node {
  public:
    Concat();
    Concat(std::string n_name, std::vector<std::string> n_inputs,
            std::vector<std::string> n_outputs,
            std::vector<int> axis);
    void fill_attributes(DataSerialization &layer_serializer) override;
  private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<int> axis;
};

} //namespace op

} //namespace metawarenn
