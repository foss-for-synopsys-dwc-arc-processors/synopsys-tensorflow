#include "batch_normalization.h"

namespace metawarenn {

namespace op {

BatchNormalization::BatchNormalization(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs) : Node(n_name, "BatchNormalization") {
  inputs = n_inputs;
  outputs = n_outputs;
  }
} //namespace op

} //namespace metawarenn
