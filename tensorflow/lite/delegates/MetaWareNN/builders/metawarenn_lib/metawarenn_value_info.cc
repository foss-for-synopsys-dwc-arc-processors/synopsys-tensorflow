#include "metawarenn_value_info.h"

namespace metawarenn {

//ONNXConstructor
#if ONNX
MWNNValueInfo::MWNNValueInfo(ValueInfoProto& onnx_value_info_proto) {
  name = onnx_value_info_proto.name();
  if(onnx_value_info_proto.type().tensor_type().has_elem_type()) {
    in_type = onnx_value_info_proto.type().tensor_type().elem_type();
    t_type = ElementType::get_mwnn_type_onnx(in_type);
    for (const auto& onnx_dim : onnx_value_info_proto.type().tensor_type().shape().dim()) {
      if (onnx_dim.has_dim_value()) {
        dims.emplace_back(onnx_dim.dim_value());
      }
      else {
        dims.emplace_back(1);
      }
    }
  }
}
#endif

#if TFLITE
//TFConstructor
MWNNValueInfo::MWNNValueInfo(std::string m_name, std::vector<int> m_dims, int m_type) {
  name = m_name;
  dims = m_dims;
  in_type = m_type;
  t_type = ElementType::get_mwnn_type_tf(in_type);
}
#endif

#if GLOW
//GLOWConstructor
MWNNValueInfo::MWNNValueInfo(std::string m_name, std::vector<int> m_dims, ElemKind m_type) {
  name = m_name;
  dims = m_dims;
  in_type = int(m_type);
  t_type = ElementType::get_mwnn_type_glow(m_type);
}
#endif

#if TVM
//TVMConstructor
MWNNValueInfo::MWNNValueInfo(std::string m_name, std::vector<int> m_dims, int m_type) {
  name = m_name;
  dims = m_dims;
  in_type = m_type;
  t_type = ElementType::get_mwnn_type_tvm(in_type);
}
#endif
} //namespace metawarenn
