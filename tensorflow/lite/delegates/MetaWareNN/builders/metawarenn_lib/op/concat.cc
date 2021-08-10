#include "concat.h"

namespace metawarenn {

namespace op {

Concat::Concat() { std::cout << "\n In Concat Constructor!!!"; }

Concat::Concat(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs,
                  std::vector<int> n_axis) : Node(n_name, "Concat") {
  inputs = n_inputs;
  outputs = n_outputs;
  axis = n_axis;
  }

void Concat::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Concat fill_attributes!!!";
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
