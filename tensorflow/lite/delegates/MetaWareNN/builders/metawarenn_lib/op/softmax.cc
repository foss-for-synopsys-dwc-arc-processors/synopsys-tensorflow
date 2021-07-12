#include "softmax.h"

namespace metawarenn {

namespace op {

Softmax::Softmax() { std::cout << "\n In Softmax Constructor!!!"; }

Softmax::Softmax(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Softmax") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Softmax::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Softmax fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
