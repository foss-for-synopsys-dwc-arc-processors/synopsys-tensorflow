#include "constant.h"

namespace metawarenn {

namespace op {

Constant::Constant(std::string n_name, std::vector<int> n_shape, std::vector<float> n_data, ElementType::element_type n_data_type)
         : Node(n_name, "Constant") {
  shape = n_shape;
  data = n_data;
  data_type = n_data_type;
  }
} //namespace op

} //namespace metawarenn
