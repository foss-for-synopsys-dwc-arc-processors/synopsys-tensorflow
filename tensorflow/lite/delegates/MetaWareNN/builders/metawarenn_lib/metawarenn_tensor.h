#ifndef METAWARENN_TENSOR_H_
#define METAWARENN_TENSOR_H_

#include "metawarenn_model.h"
#include "metawarenn_element.h"
#include "op/constant.h"

namespace metawarenn {

class MWNNTensor {
  public:
    MWNNTensor(std::string m_name, std::vector<int> m_dims, int m_tf_type, std::vector<float> m_tensor);
    std::string get_name() { return name; }
    int get_type() { return tf_type; }
    std::vector<int> get_dims() { return dims; }
    std::vector<float> get_tensor() { return tensor; }
    std::shared_ptr<op::Node> get_constant_node() {
      return std::make_shared<op::Constant>(name, dims, tensor, t_type);
    }
  private:
    std::string name;
    int tf_type;
    ElementType::element_type t_type;
    std::vector<int> dims;
    std::vector<float> tensor;
};

} //namespace metawarenn

#endif //METAWARENN_TENSOR_H_
