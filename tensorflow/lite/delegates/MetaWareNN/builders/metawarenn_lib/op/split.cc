#include "split.h"

namespace metawarenn {

namespace op {

Split::Split() { std::cout << "\n In Split Constructor!!!"; }

Split::Split(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs,
          std::vector<int> n_num_splits) : Node(n_name, "Split") {
  inputs = n_inputs;
  outputs = n_outputs;
  num_splits = n_num_splits;
  }

void Split::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Split fill_attributes!!!";
  std::cout << "\n Num Splits : ";
  for (auto ns : num_splits) {
    std::cout << ns << ", ";
  }
  auto split_len = num_splits.size();
  layer_serializer.append(static_cast<uint32_t>(split_len));
  for (auto ns : num_splits) {
    layer_serializer.append(static_cast<int32_t>(ns));
  }
  }
} //namespace op

} //namespace metawarenn
