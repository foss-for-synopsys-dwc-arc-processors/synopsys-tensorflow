#include "metawarenn_value_info.h"

namespace metawarenn {

MWNNValueInfo::MWNNValueInfo(std::string m_name, std::vector<int> m_dims, int m_tf_type) {
  name = m_name;
  dims = m_dims;
  tf_type = m_tf_type;
  t_type = ElementType::get_mwnn_type(tf_type);
  //std::cout << "\n MWNNValueInfo Name: " << name << "\n MWNNValueInfo TF Type: " << tf_type << "  MWNN Type: " << (int)t_type << "  Dims: ";
  //for (auto& it : dims) { std::cout << it << ' '; }
  }
} //namespace metawarenn
