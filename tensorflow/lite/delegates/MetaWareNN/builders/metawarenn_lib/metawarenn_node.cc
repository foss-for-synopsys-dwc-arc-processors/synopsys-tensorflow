#include "metawarenn_node.h"

namespace metawarenn {

//ONNXConstructor
MWNNNode::MWNNNode(NodeProto& onnx_node_proto) {
  node_proto = onnx_node_proto;
  name = node_proto.name();
  op_type = node_proto.op_type();
  for (auto input : node_proto.input()) {
    inputs.emplace_back(input);
  }
  for (auto output : node_proto.output()) {
    outputs.emplace_back(output);
  }
  for (auto attribute_proto : node_proto.attribute()) {
    MWNNAttribute mwnn_attribute(attribute_proto);
    mwnn_attributes.emplace_back(mwnn_attribute);
    if(mwnn_attribute.get_name() == "group")
    {
      op_type = (int)mwnn_attribute.get_data()[0] == 1 ? "Conv" : "DepthwiseConv";
    }
  }
  if(op_type == "Conv" or op_type == "DepthwiseConv") {
    MWNNAttribute mwnn_attribute("activation", {0});
    mwnn_attributes.emplace_back(mwnn_attribute);
  }
}

//TFConstructor
MWNNNode::MWNNNode(std::string m_name, std::string m_op_type, std::vector<MWNNAttribute> m_mwnn_attributes, std::vector<std::string> m_inputs, std::vector<std::string> m_outputs) {
  name = m_name;
  op_type = m_op_type;
  mwnn_attributes = m_mwnn_attributes;
  inputs = m_inputs;
  outputs = m_outputs;
}
} //namespace metawarenn
