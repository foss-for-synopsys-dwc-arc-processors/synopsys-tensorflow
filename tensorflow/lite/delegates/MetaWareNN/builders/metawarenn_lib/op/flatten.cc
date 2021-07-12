#include "flatten.h"

namespace metawarenn {

namespace op {

Flatten::Flatten() { std::cout << "\n In Flatten Constructor!!!"; }

Flatten::Flatten(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs) : Node(n_name, "Flatten") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Flatten::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Flatten fill_attributes!!!";
  }

} //namespace op

} //namespace metawarenn
