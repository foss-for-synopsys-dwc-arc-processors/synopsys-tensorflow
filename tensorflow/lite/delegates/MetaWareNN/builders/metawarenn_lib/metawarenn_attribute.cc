#include "metawarenn_attribute.h"

namespace metawarenn {

//ONNXConstructor
MWNNAttribute::MWNNAttribute(AttributeProto& onnx_attribute_proto) {
  attribute_proto = onnx_attribute_proto;
  name = attribute_proto.name();
  type = attribute_proto.type();
  set_data();
}

//TFConstructor
MWNNAttribute::MWNNAttribute(std::string m_name, std::vector<int> m_data) {
  name = m_name;
  type = AttributeProto_AttributeType_INT;
  data = m_data;
}

void MWNNAttribute::set_data() {
  switch(get_t_type()) {
    case Type::float_point:
      data.push_back(get_float());
      break;
    case Type::integer:
      data.push_back(get_int());
      break;
    case Type::string:
      string_data.push_back(get_string());
      break;
    case Type::tensor:
      std::cout << "\n AttributeProto_AttributeType_TENSOR : Exiting Code Due to Nosupport";
      exit(1);
      break;
    case Type::graph:
      std::cout << "\n AttributeProto_AttributeType_GRAPH : Exiting Code Due to Nosupport";
      exit(1);
      break;
    case Type::float_point_array:
      data = get_float_array();
      break;
    case Type::integer_array:
      data = get_integer_array();
      break;
    case Type::string_array:
      string_data = get_string_array();
      break;
    case Type::tensor_array:
      std::cout << "\n AttributeProto_AttributeType_TENSORS : Exiting Code Due to Nosupport";
      exit(1);
      break;
    case Type::graph_array:
      std::cout << "\n AttributeProto_AttributeType_GRAPHS : Exiting Code Due to Nosupport";
      exit(1);
      break;
    default:
      std::cout << "\n In Default switch case : Exiting Code Due to unsupported Data Type";
      exit(1);
      break;
  }
}
void MWNNAttribute::set_data(int m_data) {
    data.clear();
    data.push_back(m_data);
}
} //namespace metawarenn
