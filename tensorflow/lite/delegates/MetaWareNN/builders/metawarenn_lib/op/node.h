#pragma once

#include <iostream>
#include <vector>

namespace metawarenn {

namespace op {

class Node {
  public:
    Node();
    Node(std::string n_name, std::string n_type);
    std::string name;
    std::string type;
};
} //namespace op

} //namespace metawarenn
