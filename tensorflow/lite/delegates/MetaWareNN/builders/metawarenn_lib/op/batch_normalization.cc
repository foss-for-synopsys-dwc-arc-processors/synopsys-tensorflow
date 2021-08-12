#include "batch_normalization.h"

namespace metawarenn {

namespace op {

BatchNormalization::BatchNormalization() { std::cout << "\n In BatchNormalization Constructor!!!"; }

BatchNormalization::BatchNormalization(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs, std::vector<float> n_epsilon, std::vector<float> n_momentum) : Node(n_name, "BatchNormalization") {
  inputs = n_inputs;
  outputs = n_outputs;
  epsilon = n_epsilon;
  momentum = n_momentum;
  }

void BatchNormalization::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In BatchNormalization fill_attributes!!!";
  std::cout << "\n Epsilon : ";
  for (auto e : epsilon) {
    std::cout << e << ", ";
  }
  std::cout << "\n Momentum : ";
  for (auto m : momentum) {
    std::cout << m << ", ";
  }
  auto e_len = epsilon.size();
  layer_serializer.append(static_cast<uint32_t>(e_len));
  for (auto e : epsilon) {
    layer_serializer.append(static_cast<float>(e));
  }
  auto m_len =  momentum.size();
  layer_serializer.append(static_cast<uint32_t>(m_len));
  for (auto m : momentum) {
    layer_serializer.append(static_cast<float>(m));
  }
}
} //namespace op

} //namespace metawarenn
