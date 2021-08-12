#include "clip.h"

namespace metawarenn {

namespace op {

Clip::Clip() { std::cout << "\n In Clip Constructor!!!"; }

Clip::Clip(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs,
         std::vector<float> n_min, std::vector<float> n_max) : Node(n_name, "Clip") {
  inputs = n_inputs;
  outputs = n_outputs;
  min = n_min;
  max = n_max;
  }

void Clip::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Clip fill_attributes!!!";
  std::cout << "\n Min : ";
  for (auto m : min) {
    std::cout << m << ", ";
  }
  std::cout << "\n Max : ";
  for (auto m : max) {
    std::cout << m << ", ";
  }
  auto min_len = min.size();
  std::cout << "\nmin_len: " << min_len;
  layer_serializer.append(static_cast<uint32_t>(min_len));
  for (auto m : min) {
    layer_serializer.append(static_cast<float>(m));
  }
  auto max_len = max.size();
  layer_serializer.append(static_cast<uint32_t>(max_len));
  for (auto m : max) {
    layer_serializer.append(static_cast<float>(m));
  }
  }
} //namespace op

} //namespace metawarenn
