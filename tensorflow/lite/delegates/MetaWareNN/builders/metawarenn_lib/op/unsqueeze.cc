#include "unsqueeze.h"

namespace metawarenn {

namespace op {

Unsqueeze::Unsqueeze() { std::cout << "\n In Unsqueeze Constructor!!!"; }

Unsqueeze::Unsqueeze(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs,
         std::vector<int> n_axes) : Node(n_name, "Unsqueeze") {
  inputs = n_inputs;
  outputs = n_outputs;
  axes = n_axes;
  }

void Unsqueeze::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Unsqueeze fill_attributes!!!";
  std::cout << "\n Axes : ";
  for (auto a : axes) {
    std::cout << a << ", ";
  }
  auto axes_len = axes.size();
  layer_serializer.append(static_cast<uint32_t>(axes_len));
  for (auto a : axes) {
    layer_serializer.append(static_cast<int32_t>(a));
  }
  }
} //namespace op

} //namespace metawarenn
