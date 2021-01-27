#include "metawarenn_tensor.h"

namespace metawarenn {

//ONNXConstructor
MWNNTensor::MWNNTensor(TensorProto& onnx_tensor_proto) {
  tensor_proto = onnx_tensor_proto;
  name = tensor_proto.name();
  in_type = tensor_proto.data_type();
  t_type = ElementType::get_mwnn_type_onnx(in_type);
  for(auto dim : tensor_proto.dims()) {
    dims.emplace_back(dim);
  }
  set_tensor();
}

//TFConstructor
MWNNTensor::MWNNTensor(std::string m_name, std::vector<int> m_dims, int m_type, std::vector<float> m_tensor) {
    name = m_name;
    dims = m_dims;
    in_type = m_type;
    t_type = ElementType::get_mwnn_type_tf(in_type);
    tensor = m_tensor;
    for (auto& it : dims) { std::cout << it << ' '; }
}

void MWNNTensor::set_tensor() {
  switch (in_type) {
    case onnx::TensorProto_DataType_FLOAT:
      tensor = get_data<float>(tensor_proto.float_data());
      break;
    case onnx::TensorProto_DataType_INT64:
      tensor = get_data<float>(tensor_proto.int64_data());
      break;
    default:
      break;
  }
}
} //namespace metawarenn
