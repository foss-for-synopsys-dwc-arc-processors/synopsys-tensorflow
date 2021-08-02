#include "exp.h"

namespace metawarenn {

namespace op {

Exp::Exp() { std::cout << "\n In Exp Constructor!!!"; }

Exp::Exp(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Exp") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Exp::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Exp fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
