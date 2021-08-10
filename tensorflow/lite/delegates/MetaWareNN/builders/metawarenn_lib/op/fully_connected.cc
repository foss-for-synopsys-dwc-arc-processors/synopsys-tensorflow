#include "fully_connected.h"

namespace metawarenn {

namespace op {

FullyConnected::FullyConnected() { std::cout << "\n In FullyConnected Constructor!!!"; }

FullyConnected::FullyConnected(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs,
                 std::vector<int> n_asymmetric_quantize_inputs,
                 std::vector<int> n_keep_num_dims,
                 std::vector<int> n_weights_format) : Node(n_name, "FullyConnected") {
  inputs = n_inputs;
  outputs = n_outputs;
  asymmetric_quantize_inputs = n_asymmetric_quantize_inputs;
  keep_num_dims = n_keep_num_dims;
  weights_format = n_weights_format;
  }

void FullyConnected::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In FullyConnected fill_attributes!!!";
  std::cout << "\n Asymmetric quantization inputs : ";
  for (auto asym : asymmetric_quantize_inputs) {
    std::cout << asym << ", ";
  }
  std::cout << "\n Keep dims : ";
  for (auto kn : keep_num_dims) {
    std::cout << kn << ", ";
  }
  std::cout << "\n Weight format : ";
  for (auto w : weights_format) {
    std::cout << w << ", ";
  }
  auto asym_len = asymmetric_quantize_inputs.size();
  layer_serializer.append(static_cast<uint32_t>(asym_len));
  for (auto asym : asymmetric_quantize_inputs) {
    layer_serializer.append(static_cast<int32_t>(asym));
  }
  auto kn_len = keep_num_dims.size();
  layer_serializer.append(static_cast<uint32_t>(kn_len));
  for (auto kn : keep_num_dims) {
    layer_serializer.append(static_cast<int32_t>(kn));
  }
  auto wf_len = weights_format.size();
  layer_serializer.append(static_cast<uint32_t>(wf_len));
  for (auto wf : weights_format) {
    layer_serializer.append(static_cast<int32_t>(wf));
  }
  }
} //namespace op

} //namespace metawarenn
