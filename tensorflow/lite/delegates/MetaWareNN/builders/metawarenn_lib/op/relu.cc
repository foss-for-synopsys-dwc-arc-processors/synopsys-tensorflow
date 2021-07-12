#include "relu.h"

namespace metawarenn {

namespace op {

Relu::Relu() { std::cout << "\n In Relu Constructor!!!"; }

Relu::Relu(std::string n_name, std::vector<std::string> n_inputs,
           std::vector<std::string> n_outputs) : Node(n_name, "Relu") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Relu::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Relu fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
