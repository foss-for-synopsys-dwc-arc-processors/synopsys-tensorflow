#ifndef METAWARENN_ATTRIBUTE_H_
#define METAWARENN_ATTRIBUTE_H_

#include "metawarenn_model.h"

namespace metawarenn {

class MWNNAttribute {
  public:
    MWNNAttribute(AttributeProto& onnx_attribute_proto);
    MWNNAttribute(std::string m_name, std::vector<int> m_data);
    void set_data();
    void set_data(int m_data);
    std::string get_name() { return name; }
    int get_type() { return type; }
    std::vector<int> get_data() { return data; }
    std::vector<std::string> get_string_data() { return string_data; }
  private:
    AttributeProto attribute_proto;
    std::string name;
    int type;
    std::vector<int> data;
    std::vector<std::string> string_data;
    //TODO : Handle the Tensor and Graph Datatypes
  public:
    enum class Type {
      undefined = AttributeProto_AttributeType_UNDEFINED,
      float_point = AttributeProto_AttributeType_FLOAT,
      integer = AttributeProto_AttributeType_INT,
      string = AttributeProto_AttributeType_STRING,
      tensor = AttributeProto_AttributeType_TENSOR,
      graph = AttributeProto_AttributeType_GRAPH,
      float_point_array = AttributeProto_AttributeType_FLOATS,
      integer_array = AttributeProto_AttributeType_INTS,
      string_array = AttributeProto_AttributeType_STRINGS,
      tensor_array = AttributeProto_AttributeType_TENSORS,
      graph_array = AttributeProto_AttributeType_GRAPHS
    };

    Type get_t_type() { return static_cast<Type>(type); }
    float get_float() { return attribute_proto.f(); }
    int64_t get_int() { return attribute_proto.i(); }
    const std::string& get_string() { return attribute_proto.s(); }
    std::vector<int> get_float_array() {
      return {std::begin(attribute_proto.floats()),
              std::end(attribute_proto.floats())};
    }
    std::vector<int> get_integer_array() {
      return {std::begin(attribute_proto.ints()),
              std::end(attribute_proto.ints())};
    }
    std::vector<std::string> get_string_array() {
      return {std::begin(attribute_proto.strings()),
              std::end(attribute_proto.strings())};
    }
};

} //namespace metawarenn

#endif //METAWARENN_ATTRIBUTE_H_
