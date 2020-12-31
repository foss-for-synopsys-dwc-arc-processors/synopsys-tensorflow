#ifndef METAWARENN_ATTRIBUTE_H_
#define METAWARENN_ATTRIBUTE_H_

#include "metawarenn_model.h"

namespace metawarenn {

class MWNNAttribute {
  public:
    MWNNAttribute(std::string m_name, std::vector<int> m_data);
    std::string get_name() { return name; }
    std::vector<int> get_data() { return data; }
  private:
    std::string name;
    std::vector<int> data;
};

} //namespace metawarenn

#endif //METAWARENN_ATTRIBUTE_H_
