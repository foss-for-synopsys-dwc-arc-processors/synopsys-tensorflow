#ifndef METAWARENN_ATTRIBUTE_H_
#define METAWARENN_ATTRIBUTE_H_

#include "metawarenn_common.h"
namespace metawarenn {

class MWNNAttribute {
  public:
    MWNNAttribute() = default;
    #if ONNX
    MWNNAttribute(AttributeProto& onnx_attribute_proto);
    void set_data(AttributeProto& onnx_attribute_proto);
    #endif
    MWNNAttribute(std::string m_name, std::vector<int> m_data);
    void set_data(int m_data);
    std::string get_name() { return name; }
    int get_type() { return type; }
    std::vector<int> get_data() { return data; }
    std::vector<std::string> get_string_data() { return string_data; }
  private:
    std::string name;
    int type;
    std::vector<int> data;
    std::vector<std::string> string_data;
};

} //namespace metawarenn

#endif //METAWARENN_ATTRIBUTE_H_
