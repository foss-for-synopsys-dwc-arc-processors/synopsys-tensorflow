#include "sum.h"

namespace metawarenn {

namespace op {

Sum::Sum() { std::cout << "\n In Sum Constructor!!!"; }

Sum::Sum(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Sum") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Sum::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Sum fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
