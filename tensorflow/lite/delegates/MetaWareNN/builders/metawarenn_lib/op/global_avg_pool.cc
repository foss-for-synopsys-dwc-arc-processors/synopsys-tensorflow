#include "global_avg_pool.h"

namespace metawarenn {

namespace op {

GlobalAvgPool::GlobalAvgPool() { std::cout << "\n In GlobalAvgPool Constructor!!!"; }

GlobalAvgPool::GlobalAvgPool(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs) : Node(n_name, "GlobalAvgPool") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void GlobalAvgPool::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In GlobalAvgPool fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
