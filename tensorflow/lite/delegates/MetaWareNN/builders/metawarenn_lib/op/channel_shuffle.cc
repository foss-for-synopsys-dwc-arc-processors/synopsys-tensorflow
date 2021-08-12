#include "channel_shuffle.h"

namespace metawarenn {

namespace op {

ChannelShuffle::ChannelShuffle() { std::cout << "\n In ChannelShuffle Constructor!!!"; }

ChannelShuffle::ChannelShuffle(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs, std::vector<int> n_group,
         std::vector<int> n_kernel) : Node(n_name, "ChannelShuffle") {
  inputs = n_inputs;
  outputs = n_outputs;
  group = n_group;
  kernel = n_kernel;
  }

void ChannelShuffle::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In ChannelShuffle fill_attributes!!!";
  std::cout << "\n Group : ";
  for (auto g : group) {
    std::cout << g << ", ";
  }
  std::cout << "\n Kernel : ";
  for (auto k : kernel) {
    std::cout << k << ", ";
  }
  auto grp_len = group.size();
  layer_serializer.append(static_cast<uint32_t>(grp_len));
  for (auto g : group) {
    layer_serializer.append(static_cast<int32_t>(g));
  }
  auto ker_len = kernel.size();
  layer_serializer.append(static_cast<uint32_t>(ker_len));
  for (auto k : kernel) {
    layer_serializer.append(static_cast<int32_t>(k));
  }
  }
} //namespace op

} //namespace metawarenn
