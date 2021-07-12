#include "input_data.h"

namespace metawarenn {

namespace op {

InputData::InputData() { std::cout << "\n In InputData Constructor!!!"; }

InputData::InputData(std::string n_name, std::vector<int> n_shape, ElementType::element_type n_data_type)
         : Node(n_name, "InputData") {
  shape = n_shape;
  data_type = n_data_type;
  }

void InputData::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In InputData fill_attributes!!!";
  }
} //namespace op

} //namespace metawarenn
