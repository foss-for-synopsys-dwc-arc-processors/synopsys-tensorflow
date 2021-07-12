#include "pad.h"

namespace metawarenn {

namespace op {

Pad::Pad() { std::cout << "\n In Pad Constructor!!!"; }

Pad::Pad(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Pad") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Pad::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Pad fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
