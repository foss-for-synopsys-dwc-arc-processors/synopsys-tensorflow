#include "metawarenn_tensor.h"

namespace metawarenn {

MWNNTensor::MWNNTensor(std::string m_name, std::vector<int> m_dims, int m_tf_type, std::vector<float> m_tensor) {
    name = m_name;
    dims = m_dims;
    tf_type = m_tf_type;
    t_type = ElementType::get_mwnn_type(tf_type);
    tensor = m_tensor;
    std::cout << "\n MWNNTensor Name: " << name << "  MWNNTensor Dims: ";
    for (auto& it : dims) { std::cout << it << ' '; }
  }
} //namespace metawarenn
