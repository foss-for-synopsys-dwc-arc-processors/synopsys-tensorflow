#pragma once

#include "node.h"

namespace metawarenn {

namespace op {

class Gemm : public Node {
  public:
    Gemm();
    Gemm(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs,
          std::vector<int> n_transA, std::vector<int> n_transB,
          std::vector<int> n_alpha, std::vector<int> n_beta);
    void fill_attributes(DataSerialization &layer_serializer) override;
  private:
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<int> transA;
    std::vector<int> transB;
    std::vector<int> alpha;
    std::vector<int> beta;
};

} //namespace op

} //namespace metawarenn
