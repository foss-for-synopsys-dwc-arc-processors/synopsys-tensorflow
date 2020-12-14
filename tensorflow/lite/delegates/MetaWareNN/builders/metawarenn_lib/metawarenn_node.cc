#include "metawarenn_node.h"

namespace metawarenn {

MWNNNode::MWNNNode(std::string m_name, std::string m_op_type, std::vector<std::string> m_inputs, std::vector<std::string> m_outputs) {
    name = m_name;
    op_type = m_op_type;
    inputs = m_inputs;
    outputs = m_outputs;
    std::cout << "\n\n MWNNNode Name: " << name << "  Op Type: " << op_type;
  }
} //namespace metawarenn
