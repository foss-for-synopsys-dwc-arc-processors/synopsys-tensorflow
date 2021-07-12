#pragma once

#include "node.h"
#include "../metawarenn_element.h"

namespace metawarenn {

namespace op {

class Constant : public Node {
  public:
    Constant();
    Constant(std::string n_name, std::vector<int> n_shape, std::vector<float> n_data, ElementType::element_type n_data_type);
    void fill_attributes(DataSerialization &layer_serializer) override;
    std::vector<int> shape;
    std::vector<float> data;
    ElementType::element_type data_type;
};

} //namespace op

} //namespace metawarenn
