#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class Split : public Node {
  public:
    Split();
    Split(std::string n_name, std::vector<std::string> n_inputs,
        std::vector<std::string> n_outputs,
         std::vector<int> n_num_splits);
    void fill_attributes(DataSerialization &layer_serializer) override;
  private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<int> num_splits;
};

} //namespace op

} //namespace metawarenn
