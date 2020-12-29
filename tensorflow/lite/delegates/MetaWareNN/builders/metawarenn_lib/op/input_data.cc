#include "input_data.h"

namespace metawarenn {

namespace op {

InputData::InputData(std::string n_name, std::vector<int> n_shape, ElementType::element_type n_data_type)
         : Node(n_name, "InputData") {
  shape = n_shape;
  data_type = n_data_type;
  }
} //namespace op

} //namespace metawarenn
