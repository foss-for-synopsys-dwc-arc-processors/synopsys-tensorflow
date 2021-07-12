#include "transpose.h"

namespace metawarenn {

namespace op {

Transpose::Transpose() { std::cout << "\n In Transpose Constructor!!!"; }

Transpose::Transpose(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs) : Node(n_name, "Transpose") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Transpose::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Transpose fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
