#pragma once

#include <iostream>
#include <vector>
#include "../executable_network/metawarenn_serialization.h"

namespace metawarenn {

namespace op {

class Node {
  public:
    Node();
    Node(std::string n_name, std::string n_type);
    virtual ~Node();
    virtual void fill_attributes(DataSerialization &layer_serializer) = 0;
    std::string name;
    std::string type;
};
} //namespace op

} //namespace metawarenn
