#include "metawarenn_tensor.h"

namespace metawarenn {

MWNNTensor::MWNNTensor(std::string m_name, std::vector<int> m_dims, std::vector<float> m_tensor) {
    name = m_name;
    dims = m_dims;
    tensor = m_tensor;
    std::cout << "\n MWNNTensor Name: " << name << "  MWNNTensor Dims: ";
    for (auto& it : dims) { std::cout << it << ' '; }
  }
} //namespace metawarenn
