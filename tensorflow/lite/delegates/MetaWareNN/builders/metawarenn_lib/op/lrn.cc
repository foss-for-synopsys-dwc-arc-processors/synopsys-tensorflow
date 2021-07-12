#include "lrn.h"

namespace metawarenn {

namespace op {

LRN::LRN() { std::cout << "\n In LRN Constructor!!!"; }

LRN::LRN(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs) : Node(n_name, "LRN") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void LRN::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In LRN fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
