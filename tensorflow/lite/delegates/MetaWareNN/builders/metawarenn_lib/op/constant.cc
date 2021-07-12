#include "constant.h"

namespace metawarenn {

namespace op {

Constant::Constant() { std::cout << "\n In Constant Constructor!!!"; }

Constant::Constant(std::string n_name, std::vector<int> n_shape, std::vector<float> n_data, ElementType::element_type n_data_type)
         : Node(n_name, "Constant") {
  shape = n_shape;
  data = n_data;
  data_type = n_data_type;
  }

void Constant::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Constant fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
