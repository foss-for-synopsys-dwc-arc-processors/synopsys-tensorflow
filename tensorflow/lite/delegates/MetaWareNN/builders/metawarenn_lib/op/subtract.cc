#include "subtract.h"

namespace metawarenn {

namespace op {

Subtract::Subtract() { std::cout << "\n In Subtract Constructor!!!"; }

Subtract::Subtract(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Subtract") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Subtract::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Subtract fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
