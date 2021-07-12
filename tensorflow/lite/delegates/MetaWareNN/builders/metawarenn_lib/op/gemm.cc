#include "gemm.h"

namespace metawarenn {

namespace op {

Gemm::Gemm() { std::cout << "\n In Gemm Constructor!!!"; }

Gemm::Gemm(std::string n_name, std::vector<std::string> n_inputs,
           std::vector<std::string> n_outputs) : Node(n_name, "Gemm") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Gemm::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Gemm fill_attributes!!!";
  }

} //namespace op

} //namespace metawarenn
