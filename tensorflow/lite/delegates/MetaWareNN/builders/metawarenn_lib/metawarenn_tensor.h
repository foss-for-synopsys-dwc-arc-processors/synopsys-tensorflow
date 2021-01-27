#ifndef METAWARENN_TENSOR_H_
#define METAWARENN_TENSOR_H_

#include "metawarenn_model.h"
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
    MWNNTensor(TensorProto& onnx_tensor_proto);
    MWNNTensor(std::string m_name, std::vector<int> m_dims, int m_type, std::vector<float> m_tensor);
    void set_tensor();
    std::string get_name() { return name; }
    int get_type() { return in_type; }
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
    TensorProto tensor_proto;
    std::string name;
    int in_type;
    ElementType::element_type t_type;
    std::vector<int> dims;
    std::vector<float> tensor;
};

} //namespace metawarenn

#endif //METAWARENN_TENSOR_H_
