#include "divide.h"

namespace metawarenn {

namespace op {

Divide::Divide() { std::cout << "\n In Divide Constructor!!!"; }

Divide::Divide(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Divide") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Divide::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Divide fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
