#include "max.h"

namespace metawarenn {

namespace op {

Max::Max() { std::cout << "\n In Max Constructor!!!"; }

Max::Max(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Max") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Max::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Max fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
