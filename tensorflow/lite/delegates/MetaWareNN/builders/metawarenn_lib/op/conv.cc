#include "conv.h"

namespace metawarenn {

namespace op {

Conv::Conv() { std::cout << "\n In Conv Constructor!!!"; }

Conv::Conv(std::string n_name, std::vector<std::string> n_inputs,
           std::vector<std::string> n_outputs,
           std::vector<int> n_dilations,
           std::vector<int> n_strides,
           std::vector<int> n_pads,
           int n_activation) : Node(n_name, "Conv") {
  inputs = n_inputs;
  outputs = n_outputs;
  dilations = n_dilations;
  strides = n_strides;
  pads = n_pads;
  activation = n_activation;
  }

void Conv::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Conv fill_attributes!!!";
  std::cout << "\n Dilations : ";
  for (auto dil : dilations) {
    std::cout << dil << ", ";
  }
  std::cout << "\n Strides : ";
  for (auto st : strides) {
    std::cout << st << ", ";
  }
  std::cout << "\n Pads : ";
  for (auto p : pads) {
    std::cout << p << ", ";
  }
  std::cout << "\n Activation : " << activation;
  auto dil_len = dilations.size();
  layer_serializer.append(static_cast<uint32_t>(dil_len));
  for (auto dil : dilations) {
    layer_serializer.append(static_cast<int32_t>(dil));
  }
  auto str_len = strides.size();
  layer_serializer.append(static_cast<uint32_t>(str_len));
  for (auto str : strides) {
    layer_serializer.append(static_cast<int32_t>(str));
  }
  auto p_len = pads.size();
  layer_serializer.append(static_cast<uint32_t>(p_len));
  for (auto p : pads) {
    layer_serializer.append(static_cast<int32_t>(p));
  }
  layer_serializer.append(static_cast<int32_t>(activation));
  }
} //namespace op

} //namespace metawarenn
