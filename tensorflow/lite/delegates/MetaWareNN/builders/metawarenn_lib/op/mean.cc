#include "mean.h"

namespace metawarenn {

namespace op {

Mean::Mean() { std::cout << "\n In Mean Constructor!!!"; }

Mean::Mean(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs) : Node(n_name, "Mean") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Mean::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Mean fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
