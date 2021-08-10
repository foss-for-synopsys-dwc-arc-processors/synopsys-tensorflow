#include "gather.h"

namespace metawarenn {

namespace op {

Gather::Gather() { std::cout << "\n In Gather Constructor!!!"; }

Gather::Gather(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs,
         std::vector<int> n_axis) : Node(n_name, "Gather") {
  inputs = n_inputs;
  outputs = n_outputs;
  axis = n_axis;
  }

void Gather::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Gather fill_attributes!!!";
  std::cout << "\n Axis : ";
  for (auto a : axis) {
    std::cout << a << ", ";
  }
  auto axis_len = axis.size();
  layer_serializer.append(static_cast<uint32_t>(axis_len));
  for (auto a : axis) {
    layer_serializer.append(static_cast<int32_t>(a));
  }
  }
} //namespace op

} //namespace metawarenn
