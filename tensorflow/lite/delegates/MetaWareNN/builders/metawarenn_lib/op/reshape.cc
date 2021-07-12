#include "reshape.h"

namespace metawarenn {

namespace op {

Reshape::Reshape() { std::cout << "\n In Reshape Constructor!!!"; }

Reshape::Reshape(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs) : Node(n_name, "Reshape") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Reshape::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Reshape fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
