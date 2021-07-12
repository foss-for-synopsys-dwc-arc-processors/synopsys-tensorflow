#include "shape.h"

namespace metawarenn {

namespace op {

Shape::Shape() { std::cout << "\n In Shape Constructor!!!"; }

Shape::Shape(std::string n_name, std::vector<std::string> n_inputs,
         std::vector<std::string> n_outputs) : Node(n_name, "Shape") {
  inputs = n_inputs;
  outputs = n_outputs;
  }

void Shape::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Shape fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
