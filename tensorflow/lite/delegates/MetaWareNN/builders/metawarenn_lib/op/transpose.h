#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class Transpose : public Node {
  public:
    Transpose(std::string n_name, std::vector<std::string> n_inputs,
            std::vector<std::string> n_outputs);
  private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
};

} //namespace op

} //namespace metawarenn