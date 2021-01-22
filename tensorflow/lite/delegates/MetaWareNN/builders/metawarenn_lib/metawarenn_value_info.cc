#include "metawarenn_value_info.h"

namespace metawarenn {

MWNNValueInfo::MWNNValueInfo(std::string m_name, std::vector<int> m_dims, int m_type) {
  name = m_name;
  dims = m_dims;
  in_type = m_type;
  t_type = ElementType::get_mwnn_type_tf(in_type);
}
} //namespace metawarenn
