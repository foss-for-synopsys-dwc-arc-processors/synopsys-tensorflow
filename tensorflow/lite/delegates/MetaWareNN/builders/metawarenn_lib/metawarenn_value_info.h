#ifndef METAWARENN_VALUE_INFO_H_
#define METAWARENN_VALUE_INFO_H_

#include "metawarenn_model.h"
#include "metawarenn_element.h"
#include "op/input_data.h"

namespace metawarenn {

class MWNNValueInfo {
  public:
    MWNNValueInfo() = default;
    MWNNValueInfo(ValueInfoProto& onnx_value_info_proto);
    MWNNValueInfo(std::string m_name, std::vector<int> m_dims, int m_type);
    std::string get_name() { return name; }
    int get_type() { return in_type; }
    std::vector<int> get_dims() { return dims; }
    std::shared_ptr<op::Node> get_node() {
      return std::make_shared<op::InputData>(name, dims, t_type);
    }
    void update_dims(std::vector<int> n_dims) {
      dims = n_dims;
    }
  private:
    ValueInfoProto value_info_proto;
    std::string name;
    int in_type;
    ElementType::element_type t_type;
    std::vector<int> dims;
};

} //namespace metawarenn

#endif //METAWARENN_VALUE_INFO_H_
