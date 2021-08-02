#include "minimum.h"

namespace metawarenn {

namespace op {

Minimum::Minimum() { std::cout << "\n In Minimum Constructor!!!"; }

Minimum::Minimum(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Minimum") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Minimum::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Add fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
