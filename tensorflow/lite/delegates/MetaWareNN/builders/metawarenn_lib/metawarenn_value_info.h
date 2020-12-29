#ifndef METAWARENN_VALUE_INFO_H_
#define METAWARENN_VALUE_INFO_H_

#include "metawarenn_model.h"
#include "metawarenn_element.h"

namespace metawarenn {

class MWNNValueInfo {
  public:
    MWNNValueInfo(std::string m_name, std::vector<int> m_dims, int m_tf_type);
    std::string get_name() { return name; }
    int get_type() { return tf_type; }
    std::vector<int> get_dims() { return dims; }
  private:
    std::string name;
    int tf_type;
    ElementType::element_type t_type;
    std::vector<int> dims;
};

} //namespace metawarenn

#endif //METAWARENN_VALUE_INFO_H_
