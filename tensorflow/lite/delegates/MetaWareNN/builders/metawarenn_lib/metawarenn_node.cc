#include "metawarenn_node.h"

namespace metawarenn {

MWNNNode::MWNNNode(std::string m_name, std::string m_op_type, std::vector<MWNNAttribute> m_mwnn_attributes, std::vector<std::string> m_inputs, std::vector<std::string> m_outputs) {
  name = m_name;
  op_type = m_op_type;
  mwnn_attributes = m_mwnn_attributes;
  inputs = m_inputs;
  outputs = m_outputs;
}
} //namespace metawarenn
