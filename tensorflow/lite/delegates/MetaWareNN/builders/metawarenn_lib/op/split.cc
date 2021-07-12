#include "split.h"

namespace metawarenn {

namespace op {

Split::Split() { std::cout << "\n In Split Constructor!!!"; }

Split::Split(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Split") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Split::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Split fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
