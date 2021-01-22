#include "metawarenn_tensor.h"

namespace metawarenn {

MWNNTensor::MWNNTensor(std::string m_name, std::vector<int> m_dims, int m_type, std::vector<float> m_tensor) {
    name = m_name;
    dims = m_dims;
    in_type = m_type;
    t_type = ElementType::get_mwnn_type_tf(in_type);
    tensor = m_tensor;
    for (auto& it : dims) { std::cout << it << ' '; }
  }
} //namespace metawarenn
