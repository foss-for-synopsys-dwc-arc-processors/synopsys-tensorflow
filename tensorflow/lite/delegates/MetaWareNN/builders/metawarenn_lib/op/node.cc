#include "node.h"

namespace metawarenn {

namespace op {

Node::Node() { /*std::cout << "\n Constructor Base Node Class";*/ }

Node::Node(std::string n_name, std::string n_type) {
  name = n_name;
  type = n_type;
  //std::cout << "\n Name : " << name << " Type : " << type;
  }

void Node::fill_attributes(DataSerialization &layer_serializer) {
  std::cout << "\n In Base Node fill_attributes!!!";
  }

Node::~Node() { /*std::cout << "\n Destructor Base Node Class";*/ }
} //namespace op

} //namespace metawarenn
