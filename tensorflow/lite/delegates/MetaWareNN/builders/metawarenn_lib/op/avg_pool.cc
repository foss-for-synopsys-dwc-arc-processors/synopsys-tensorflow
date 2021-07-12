#include "avg_pool.h"

namespace metawarenn {

namespace op {

AvgPool::AvgPool() { std::cout << "\n In AvgPool Constructor!!!"; }

AvgPool::AvgPool(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs) : Node(n_name, "AvgPool") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void AvgPool::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In AvgPool fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
