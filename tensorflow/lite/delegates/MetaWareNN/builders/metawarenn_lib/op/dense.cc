#include "dense.h"

namespace metawarenn {

namespace op {

Dense::Dense() { std::cout << "\n In Dense Constructor!!!"; }

Dense::Dense(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs): Node(n_name, "Dense") {
  inputs = n_inputs;
  outputs = n_outputs;
  //units = n_units; //attribute only available for specific model
  }

void Dense::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Dense fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
