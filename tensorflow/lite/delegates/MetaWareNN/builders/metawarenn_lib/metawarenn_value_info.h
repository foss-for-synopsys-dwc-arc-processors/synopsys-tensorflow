#ifndef METAWARENN_VALUE_INFO_H_
#define METAWARENN_VALUE_INFO_H_

#include "metawarenn_common.h"
#include "metawarenn_element.h"
#include "op/input_data.h"

namespace metawarenn {

class MWNNValueInfo {
  public:
    MWNNValueInfo() = default;
    #if ONNX
    MWNNValueInfo(ValueInfoProto& onnx_value_info_proto);
    #endif
    #if TFLITE
    MWNNValueInfo(std::string m_name, std::vector<int> m_dims, int m_type);
    #endif
    #if GLOW
    MWNNValueInfo(std::string m_name, std::vector<int> m_dims, ElemKind m_type);
    #endif
    std::string get_name() { return name; }
    int get_type() { return int(t_type); }
    std::vector<int> get_dims() { return dims; }
    std::shared_ptr<op::Node> get_node() {
      return std::make_shared<op::InputData>(name, dims, t_type);
    }
    void update_dims(std::vector<int> n_dims) {
      dims = n_dims;
    }
  private:
    std::string name;
    int in_type;
    ElementType::element_type t_type;
    std::vector<int> dims;
};

} //namespace metawarenn

#endif //METAWARENN_VALUE_INFO_H_
