#include "avg_pool.h"

namespace metawarenn {

namespace op {

AvgPool::AvgPool() { std::cout << "\n In AvgPool Constructor!!!"; }

AvgPool::AvgPool(std::string n_name, std::vector<std::string> n_inputs,
                std::vector<std::string> n_outputs,
                std::vector<int> n_kernel_shape,
                std::vector<int> n_strides,
                std::vector<int> n_pads) : Node(n_name, "AveragePool") {
  inputs = n_inputs;
  outputs = n_outputs;
  kernel_shape = kernel_shape;
  strides = n_strides;
  pads = n_pads;
  }

void AvgPool::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In AvgPool fill_attributes!!!";
  std::cout << "\n Pool Size : ";
  for (auto psize : kernel_shape) {
    std::cout << psize << ", ";
  }
  std::cout << "\n Strides : ";
  for (auto st : strides) {
    std::cout << st << ", ";
  }
  std::cout << "\n Pads : ";
  for (auto p : pads) {
    std::cout << p << ", ";
  }
  auto psize_len = kernel_shape.size();
  layer_serializer.append(static_cast<uint32_t>(psize_len));
  for (auto psize : kernel_shape) {
    layer_serializer.append(static_cast<int32_t>(psize));
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
  }
} //namespace op

} //namespace metawarenn
