#include "metawarenn_attribute.h"

namespace metawarenn {

MWNNAttribute::MWNNAttribute(std::string m_name, std::vector<int> m_data) {
  name = m_name;
  data = m_data;
  //std::cout << "\n In MWNNAttribute : Name : " << name << "  Data : ";
  //for (auto& it : data) { std::cout << it << ' '; };
}
} //namespace metawarenn
