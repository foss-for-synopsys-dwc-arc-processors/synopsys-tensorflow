#include "unsqueeze.h"

namespace metawarenn {

namespace op {

Unsqueeze::Unsqueeze() { std::cout << "\n In Unsqueeze Constructor!!!"; }

Unsqueeze::Unsqueeze(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Unsqueeze") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Unsqueeze::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Unsqueeze fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
