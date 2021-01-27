#include "metawarenn_value_info.h"

namespace metawarenn {

//ONNXConstructor
MWNNValueInfo::MWNNValueInfo(ValueInfoProto& onnx_value_info_proto) {
  value_info_proto = onnx_value_info_proto;
  name = value_info_proto.name();
  if(value_info_proto.type().tensor_type().has_elem_type()) {
    in_type = value_info_proto.type().tensor_type().elem_type();
    t_type = ElementType::get_mwnn_type_onnx(in_type);
    for (const auto& onnx_dim : value_info_proto.type().tensor_type().shape().dim()) {
      if (onnx_dim.has_dim_value()) {
        dims.emplace_back(onnx_dim.dim_value());
      }
      else {
        dims.emplace_back(0);
      }
    }
  }
}

//TFConstructor
MWNNValueInfo::MWNNValueInfo(std::string m_name, std::vector<int> m_dims, int m_type) {
  name = m_name;
  dims = m_dims;
  in_type = m_type;
  t_type = ElementType::get_mwnn_type_tf(in_type);
}
} //namespace metawarenn
