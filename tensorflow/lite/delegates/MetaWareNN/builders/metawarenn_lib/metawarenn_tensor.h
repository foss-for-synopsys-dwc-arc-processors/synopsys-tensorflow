#ifndef METAWARENN_TENSOR_H_
#define METAWARENN_TENSOR_H_

#include "metawarenn_common.h"
#include "metawarenn_element.h"
#include "op/constant.h"

namespace metawarenn {

 template <typename T, typename Container>
inline std::vector<T> get_data(const Container& container)
{
  return std::vector<T>(std::begin(container), std::end(container));
}

class MWNNTensor {
  public:
    MWNNTensor() = default;
    #if ONNX
    MWNNTensor(TensorProto& onnx_tensor_proto);
    void set_tensor(TensorProto& onnx_tensor_proto);
    #endif
    #if TFLITE
    MWNNTensor(std::string m_name, std::vector<int> m_dims, int m_type, std::vector<float> m_tensor);
    #endif
    #if GLOW
    MWNNTensor(std::string m_name, std::vector<int> m_dims, ElemKind m_type, std::vector<float> m_tensor);
    #endif
    #if TVM
    MWNNTensor(std::string m_name, std::vector<int> m_dims, int m_type, std::vector<float> m_tensor);
    #endif
    MWNNTensor(std::string t_name, int type, std::vector<int> t_shape);
    std::string get_name() { return name; }
    int get_type() { return int(t_type); }
    void set_index(uint32_t value) { index = value; }
    void set_offset(uint32_t value) { offset = value; }
    uint32_t get_index() { return index; }
    uint32_t get_offset() { return offset; }
    std::vector<int> get_dims() { return dims; }
    std::vector<float> get_tensor() { return tensor; }
    std::shared_ptr<op::Node> get_constant_node() {
      return std::make_shared<op::Constant>(name, dims, tensor, t_type);
    }
    void update_tensor(std::vector<int> n_dims, std::vector<float> n_tensor) {
      dims = n_dims;
      tensor = n_tensor;
    }
  private:
    std::string name;
    uint32_t index = 1; //used to maintain the constant initializers order index
    uint32_t offset = 0; //used to maintain the constant initializers binary size
    int in_type;
    ElementType::element_type t_type;
    std::vector<int> dims;
    std::vector<float> tensor;
};

} //namespace metawarenn

#endif //METAWARENN_TENSOR_H_
