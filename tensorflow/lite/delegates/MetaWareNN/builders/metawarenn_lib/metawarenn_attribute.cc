#include "metawarenn_attribute.h"

namespace metawarenn {


//ONNXConstructor
#if ONNX
MWNNAttribute::MWNNAttribute(AttributeProto& onnx_attribute_proto) {
  name = onnx_attribute_proto.name();
  type = (int)ElementType::get_mwnn_attr_type_onnx(onnx_attribute_proto.type());
  set_data(onnx_attribute_proto);
}

void MWNNAttribute::set_data(AttributeProto& onnx_attribute_proto) {
  switch(onnx_attribute_proto.type()) {
    case AttributeProto_AttributeType_FLOAT:
      float_data.push_back(onnx_attribute_proto.f());
      break;
    case AttributeProto_AttributeType_INT:
      int_data.push_back(onnx_attribute_proto.i());
      break;
    case AttributeProto_AttributeType_STRING:
      string_data.push_back(onnx_attribute_proto.s());
      break;
    case AttributeProto_AttributeType_TENSOR:
      std::cout << "\n AttributeProto_AttributeType_TENSOR : Exiting Code Due to Nosupport";
      exit(1);
      break;
    case AttributeProto_AttributeType_GRAPH:
      std::cout << "\n AttributeProto_AttributeType_GRAPH : Exiting Code Due to Nosupport";
      exit(1);
      break;
    case AttributeProto_AttributeType_FLOATS:
      float_data.assign(std::begin(onnx_attribute_proto.floats()), std::end(onnx_attribute_proto.floats()));
      break;
    case AttributeProto_AttributeType_INTS:
      int_data.assign(std::begin(onnx_attribute_proto.ints()), std::end(onnx_attribute_proto.ints()));
      break;
    case AttributeProto_AttributeType_STRINGS:
      string_data.assign(std::begin(onnx_attribute_proto.strings()), std::end(onnx_attribute_proto.strings()));
      break;
    case AttributeProto_AttributeType_TENSORS:
      std::cout << "\n AttributeProto_AttributeType_TENSORS : Exiting Code Due to Nosupport";
      exit(1);
      break;
    case AttributeProto_AttributeType_GRAPHS:
      std::cout << "\n AttributeProto_AttributeType_GRAPHS : Exiting Code Due to Nosupport";
      exit(1);
      break;
    default:
      std::cout << "\n In Default switch case : Exiting Code Due to unsupported Data Type";
      exit(1);
      break;
  }
}
#endif

//TFConstructor, GLOWConstructor & TVMConstructor
MWNNAttribute::MWNNAttribute(std::string m_name, std::vector<int> m_data) {
  name = m_name;
  int_data = m_data;
  type = 6;
}

//TVMConstructor
MWNNAttribute::MWNNAttribute(std::string m_name, std::vector<float> m_data) {
  name = m_name;
  float_data = m_data;
  type = 3;
}

void MWNNAttribute::set_data(int m_data) {
    int_data.clear();
    int_data.push_back(m_data);
}
} //namespace metawarenn
