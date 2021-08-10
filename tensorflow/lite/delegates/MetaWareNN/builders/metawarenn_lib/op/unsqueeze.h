#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class Unsqueeze : public Node {
  public:
    Unsqueeze();
    Unsqueeze(std::string n_name, std::vector<std::string> n_inputs,
        std::vector<std::string> n_outputs,
        std::vector<int> n_axes);
    void fill_attributes(DataSerialization &layer_serializer) override;
  private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<int> axes;
};

} //namespace op

} //namespace metawarenn
