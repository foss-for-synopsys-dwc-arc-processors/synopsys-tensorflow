#include "node.h"

namespace metawarenn {

namespace op {

Node::Node() { }
Node::Node(std::string n_name, std::string n_type) {
  name = n_name;
  type = n_type;
  std::cout << "\n Name : " << name << " Type : " << type;
  }

} //namespace op

} //namespace metawarenn
