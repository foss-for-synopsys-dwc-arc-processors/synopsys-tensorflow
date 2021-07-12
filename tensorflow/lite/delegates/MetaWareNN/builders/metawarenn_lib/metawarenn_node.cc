#include "metawarenn_node.h"

namespace metawarenn {

//ONNXConstructor
#if ONNX
MWNNNode::MWNNNode(NodeProto& onnx_node_proto) {
  name = onnx_node_proto.name();
  // Creates node name if node name from loaded graph is NULL.
  if(name == "")
  {
    static int node_counter = 0;
    char* node_name = (char*)malloc(100*sizeof(char));
    std::strcpy(node_name, onnx_node_proto.op_type().c_str());
    std::strcat(node_name, std::to_string(node_counter).c_str());
    name = node_name;
    node_counter++;
  }
  op_type = onnx_node_proto.op_type();
  for (auto input : onnx_node_proto.input()) {
    inputs.emplace_back(input);
  }
  for (auto output : onnx_node_proto.output()) {
    outputs.emplace_back(output);
  }
  for (auto attribute_proto : onnx_node_proto.attribute()) {
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
#endif

//TFConstructor & GLOWConstructor
MWNNNode::MWNNNode(std::string m_name, std::string m_op_type, std::vector<MWNNAttribute> m_mwnn_attributes, std::vector<std::string> m_inputs, std::vector<std::string> m_outputs) {
  name = m_name;
  op_type = m_op_type;
  mwnn_attributes = m_mwnn_attributes;
  inputs = m_inputs;
  outputs = m_outputs;
}
} //namespace metawarenn
