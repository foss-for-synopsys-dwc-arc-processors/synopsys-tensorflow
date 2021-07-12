#include "squeeze.h"

namespace metawarenn {

namespace op {

Squeeze::Squeeze() { std::cout << "\n In Squeeze Constructor!!!"; }

Squeeze::Squeeze(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Squeeze") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Squeeze::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Squeeze fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
