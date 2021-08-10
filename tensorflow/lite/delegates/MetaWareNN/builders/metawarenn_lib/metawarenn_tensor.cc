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

void MWNNTensor::set_tensor(TensorProto& onnx_tensor_proto) {
  switch (in_type) {
    case onnx::TensorProto_DataType_FLOAT: {
      // Checks for valid float data from tensor proto
      if(onnx_tensor_proto.float_data().size() > 0)
          tensor = get_data<float>(onnx_tensor_proto.float_data());
      // Copy from float based raw data and memcpy the bytes to MWNN vector
      else if(onnx_tensor_proto.has_raw_data()) {
        auto raw_data = onnx_tensor_proto.raw_data();
        char* bytes = const_cast<char*>(onnx_tensor_proto.raw_data().c_str());
        const size_t raw_data_size = raw_data.size();
        tensor.resize(raw_data_size / sizeof(float));
        memcpy(reinterpret_cast<char*>(tensor.data()), bytes, raw_data_size);
      }
      break;
    }
    case onnx::TensorProto_DataType_INT64: {
      // Checks for valid int data from tensor proto
      if(onnx_tensor_proto.int64_data().size() > 0)
        tensor = get_data<float>(onnx_tensor_proto.int64_data());
      // Copy from int based raw data and memcpy the bytes to MWNN vector
      else if(onnx_tensor_proto.has_raw_data()) {
        auto raw_data = onnx_tensor_proto.raw_data();
        char* bytes = const_cast<char*>(onnx_tensor_proto.raw_data().c_str());
        const size_t raw_data_size = raw_data.size();
        std::vector<int64_t> int_vector(raw_data_size / sizeof(int64_t));
        memcpy(reinterpret_cast<char*>(int_vector.data()), bytes, raw_data_size);
        for (int i = 0; i < int_vector.size(); i++)
            tensor.emplace_back((float)int_vector.data()[i]);
      }
      break;
    }
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

#if TVM
//TVMConstructor
MWNNTensor::MWNNTensor(std::string m_name, std::vector<int> m_dims, int m_type, std::vector<float> m_tensor) {
    name = m_name;
    dims = m_dims;
    in_type = m_type;
    t_type = ElementType::get_mwnn_type_tvm(in_type);
    tensor = m_tensor;
}
#endif

MWNNTensor::MWNNTensor(std::string t_name, int type, std::vector<int> t_shape) {
  name = t_name;
  t_type = (ElementType::element_type)type;
  dims = t_shape;
}
} //namespace metawarenn
