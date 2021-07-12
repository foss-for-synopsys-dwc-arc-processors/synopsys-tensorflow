#include "concat.h"

namespace metawarenn {

namespace op {

Concat::Concat() { std::cout << "\n In Concat Constructor!!!"; }

Concat::Concat(std::string n_name, std::vector<std::string> n_inputs,
                 std::vector<std::string> n_outputs) : Node(n_name, "Concat") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Concat::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Concat fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
