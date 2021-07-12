#include "add.h"

namespace metawarenn {

namespace op {

Add::Add() { std::cout << "\n In Add Constructor!!!"; }

Add::Add(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Add") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Add::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Add fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
