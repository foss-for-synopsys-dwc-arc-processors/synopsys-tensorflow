#include "gather.h"

namespace metawarenn {

namespace op {

Gather::Gather() { std::cout << "\n In Gather Constructor!!!"; }

Gather::Gather(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Gather") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Gather::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Gather fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
