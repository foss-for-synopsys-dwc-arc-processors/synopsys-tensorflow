#include "mul.h"

namespace metawarenn {

namespace op {

Mul::Mul() { std::cout << "\n In Mul Constructor!!!"; }

Mul::Mul(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Mul") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Mul::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Mul fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
