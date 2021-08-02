#include "batch_flatten.h"

namespace metawarenn {

namespace op {

BatchFlatten::BatchFlatten() { std::cout << "\n In BatchFlatten Constructor!!!"; }

BatchFlatten::BatchFlatten(std::string n_name, std::vector<std::string> n_inputs,
                std::vector<std::string> n_outputs) : Node(n_name, "BatchFlatten") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void BatchFlatten::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In BatchFlatten fill_attributes!!!";
}

} //namespace op

} //namespace metawarenn
