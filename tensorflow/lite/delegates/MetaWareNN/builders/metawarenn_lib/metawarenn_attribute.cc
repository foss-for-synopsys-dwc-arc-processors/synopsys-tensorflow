#include "metawarenn_attribute.h"

namespace metawarenn {

MWNNAttribute::MWNNAttribute(std::string m_name, std::vector<int> m_data) {
  name = m_name;
  data = m_data;
  //std::cout << "\n In MWNNAttribute : Name : " << name << "  Data : ";
  //for (auto& it : data) { std::cout << it << ' '; };
}
void MWNNAttribute::set_data(int m_data) {
    data.clear();
    data.push_back(m_data);
}
} //namespace metawarenn
