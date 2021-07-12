#include "metawarenn_tensor.h"

namespace metawarenn {

//ONNXConstructor
#if ONNX
MWNNTensor::MWNNTensor(TensorProto& onnx_tensor_proto) {
  name = onnx_tensor_proto.name();
  in_type = onnx_tensor_proto.data_type();
  t_type = ElementType::get_mwnn_type_onnx(in_type);
  for(auto dim : onnx_tensor_proto.dims()) {
    dims.emplace_back(dim);
  }
  set_tensor(onnx_tensor_proto);
}

MWNNTensor::MWNNTensor(std::string t_name, std::vector<int> t_shape) {
  name = t_name;
  dims = t_shape;
}

void MWNNTensor::set_tensor(TensorProto& onnx_tensor_proto) {
  switch (in_type) {
    case onnx::TensorProto_DataType_FLOAT:
      tensor = get_data<float>(onnx_tensor_proto.float_data());
      break;
    case onnx::TensorProto_DataType_INT64:
      tensor = get_data<float>(onnx_tensor_proto.int64_data());
      break;
    default:
      break;
  }
}
#endif

#if TFLITE
//TFConstructor
MWNNTensor::MWNNTensor(std::string m_name, std::vector<int> m_dims, int m_type, std::vector<float> m_tensor) {
    name = m_name;
    dims = m_dims;
    in_type = m_type;
    t_type = ElementType::get_mwnn_type_tf(in_type);
    tensor = m_tensor;
    for (auto& it : dims) { std::cout << it << ' '; }
}
#endif

#if GLOW
//GlowConstructor
MWNNTensor::MWNNTensor(std::string m_name, std::vector<int> m_dims, ElemKind m_type, std::vector<float> m_tensor) {
    name = m_name;
    dims = m_dims;
    t_type = ElementType::get_mwnn_type_glow(m_type);
    tensor = m_tensor;
    for (auto& it : dims) { std::cout << it << ' '; }
}
#endif

MWNNTensor::MWNNTensor(std::string t_name, int type, std::vector<int> t_shape) {
  name = t_name;
  t_type = (ElementType::element_type)type;
  dims = t_shape;
}
} //namespace metawarenn
