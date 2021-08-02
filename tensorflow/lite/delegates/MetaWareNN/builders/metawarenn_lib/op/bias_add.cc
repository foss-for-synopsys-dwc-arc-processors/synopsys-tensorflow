#include "bias_add.h"

namespace metawarenn {

namespace op {

BiasAdd::BiasAdd() { std::cout << "\n In BiasAdd Constructor!!!"; }

BiasAdd::BiasAdd(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "BiasAdd") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void BiasAdd::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In BiasAdd fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
