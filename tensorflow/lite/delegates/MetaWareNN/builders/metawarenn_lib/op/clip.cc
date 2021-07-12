#include "clip.h"

namespace metawarenn {

namespace op {

Clip::Clip() { std::cout << "\n In Clip Constructor!!!"; }

Clip::Clip(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Clip") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Clip::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Clip fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
