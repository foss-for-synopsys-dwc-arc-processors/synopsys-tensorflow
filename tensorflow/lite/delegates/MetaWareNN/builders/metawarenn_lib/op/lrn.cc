#include "lrn.h"

namespace metawarenn {

namespace op {

LRN::LRN() { std::cout << "\n In LRN Constructor!!!"; }

LRN::LRN(std::string n_name, std::vector<std::string> n_inputs,
                  std::vector<std::string> n_outputs,
                  std::vector<int> n_alpha,
                  std::vector<int> n_beta,
                  std::vector<int> n_axis,
                  std::vector<int> n_size,
                  std::vector<int> n_bias, std::vector<int> n_half_window_size) : Node(n_name, "LRN") {
  inputs = n_inputs;
  outputs = n_outputs;
  alpha = n_alpha;
  beta = n_beta;
  axis = n_axis;
  size = n_size;
  bias = n_bias;
  half_window_size = n_half_window_size;
  }

void LRN::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In LRN fill_attributes!!!";
  std::cout << "\n Alpha : ";
  for (auto a : alpha) {
    std::cout << a << ", ";
  }
  std::cout << "\n Beta : ";
  for (auto b : beta) {
    std::cout << b << ", ";
  }
  std::cout << "\n Axis : ";
  for (auto a : axis) {
    std::cout << a << ", ";
  }
  std::cout << "\n Size : ";
  for (auto s : size) {
    std::cout << s << ", ";
  }
  std::cout << "\n Bias : ";
  for (auto b : bias) {
    std::cout << b << ", ";
  }
  std::cout << "\n Half window size : ";
  for (auto h : half_window_size) {
    std::cout << h << ", ";
  }
  auto alpha_size = alpha.size();
  layer_serializer.append(static_cast<uint32_t>(alpha_size));
  for (auto a : alpha) {
    layer_serializer.append(static_cast<int32_t>(a));
  }
  auto beta_len = beta.size();
  layer_serializer.append(static_cast<uint32_t>(beta_len));
  for (auto b : beta) {
    layer_serializer.append(static_cast<int32_t>(b));
  }
  auto axis_len = axis.size();
  layer_serializer.append(static_cast<uint32_t>(axis_len));
  for (auto a : axis) {
    layer_serializer.append(static_cast<int32_t>(a));
  }
  auto size_len = size.size();
  layer_serializer.append(static_cast<uint32_t>(size_len));
  for (auto s : size) {
    layer_serializer.append(static_cast<int32_t>(s));
  }
  auto bias_len = bias.size();
  layer_serializer.append(static_cast<uint32_t>(bias_len));
  for (auto b : bias) {
    layer_serializer.append(static_cast<int32_t>(b));
  }
  auto h_len = half_window_size.size();
  layer_serializer.append(static_cast<uint32_t>(h_len));
  for (auto h : half_window_size) {
    layer_serializer.append(static_cast<int32_t>(h));
  }
  }
} //namespace op

} //namespace metawarenn
