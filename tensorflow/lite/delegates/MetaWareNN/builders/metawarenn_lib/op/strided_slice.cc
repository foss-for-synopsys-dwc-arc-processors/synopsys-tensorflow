#include "strided_slice.h"

namespace metawarenn {

namespace op {

StridedSlice::StridedSlice() { std::cout << "\n In StridedSlice Constructor!!!"; }

StridedSlice::StridedSlice(std::string n_name, std::vector<std::string> n_inputs,
        std::vector<std::string> n_outputs,  std::vector<int> n_begin_mask,
        std::vector<int> n_ellipsis_mask, std::vector<int> n_end_mask,
        std::vector<int> n_new_axis_mask, std::vector<int> n_shrink_axis_mask) : Node(n_name, "StridedSlice") {
  inputs = n_inputs;
  outputs = n_outputs;
  begin_mask = n_begin_mask;
  ellipsis_mask = n_ellipsis_mask;
  end_mask = n_end_mask;
  new_axis_mask = n_new_axis_mask;
  shrink_axis_mask = n_shrink_axis_mask;
  }

void StridedSlice::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In StridedSlice fill_attributes!!!";
  std::cout << "\n Begin mask : ";
  for (auto b : begin_mask) {
    std::cout << b << ", ";
  }
  std::cout << "\n Ellipsis mask : ";
  for (auto e : ellipsis_mask) {
    std::cout << e << ", ";
  }
  std::cout << "\n End mask : ";
  for (auto e : end_mask) {
    std::cout << e << ", ";
  }
  std::cout << "\n New axis mask : ";
  for (auto n : new_axis_mask) {
    std::cout << n << ", ";
  }
  std::cout << "\n Shrink axis mask : ";
  for (auto s : shrink_axis_mask) {
    std::cout << s << ", ";
  }
  auto bm_len = begin_mask.size();
  layer_serializer.append(static_cast<uint32_t>(bm_len));
  for (auto bm : begin_mask) {
    layer_serializer.append(static_cast<int32_t>(bm));
  }
  auto e_len = ellipsis_mask.size();
  layer_serializer.append(static_cast<uint32_t>(e_len));
  for (auto em : ellipsis_mask) {
    layer_serializer.append(static_cast<int32_t>(em));
  }
  auto em_len = end_mask.size();
  layer_serializer.append(static_cast<uint32_t>(em_len));
  for (auto em : end_mask) {
    layer_serializer.append(static_cast<int32_t>(em));
  }
  auto na_len = new_axis_mask.size();
  layer_serializer.append(static_cast<uint32_t>(na_len));
  for (auto nam : new_axis_mask) {
    layer_serializer.append(static_cast<int32_t>(nam));
  }
  auto sa_len = shrink_axis_mask.size();
  layer_serializer.append(static_cast<uint32_t>(sa_len));
  for (auto sam : shrink_axis_mask) {
    layer_serializer.append(static_cast<int32_t>(sam));
  }
  }
} //namespace op

} //namespace metawarenn
