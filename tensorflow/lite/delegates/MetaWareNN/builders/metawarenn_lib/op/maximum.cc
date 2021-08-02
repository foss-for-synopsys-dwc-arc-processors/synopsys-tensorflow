#include "maximum.h"

namespace metawarenn {

namespace op {

Maximum::Maximum() { std::cout << "\n In Maximum Constructor!!!"; }

Maximum::Maximum(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Maximum") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Maximum::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Maximum fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
