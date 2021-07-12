#ifndef METAWARENN_NODE_H_
#define METAWARENN_NODE_H_

#include "metawarenn_attribute.h"
#include "op/add.h"
#include "op/conv.h"
#include "op/relu.h"
#include "op/reshape.h"
#include "op/softmax.h"
#include "op/global_avg_pool.h"
#include "op/depthwise_conv.h"
#include "op/transpose.h"
#include "op/gemm.h"
#include "op/max_pool.h"
#include "op/flatten.h"
#include "op/batch_normalization.h"
#include "op/concat.h"
#include "op/avg_pool.h"
#include "op/lrn.h"
#include "op/mul.h"
#include "op/clip.h"
#include "op/squeeze.h"
#include "op/unsqueeze.h"
#include "op/gather.h"
#include "op/shape.h"
#include "op/split.h"
#include "op/strided_slice.h"
#include "op/pad.h"
#include "op/mean.h"
#include "op/channel_shuffle.h"
#include "op/fully_connected.h"

namespace metawarenn {

class MWNNNode {
  public:
    MWNNNode() = default;
    #if ONNX
    MWNNNode(NodeProto& onnx_node_proto);
    #endif
    MWNNNode(std::string m_name, std::string m_op_type, std::vector<MWNNAttribute> m_mwnn_attributes, std::vector<std::string> m_inputs,  std::vector<std::string> m_outputs);
    std::string get_name() { return name; }
    std::string get_op_type() { return op_type; }
    std::vector<std::string> get_inputs() { return inputs; }
    std::vector<std::string> get_outputs() { return outputs; }
    void set_inputs(std::string name, int index) { inputs[index] = name; }
    void set_outputs(std::string name, int index) { outputs[index] = name; }
    std::vector<MWNNAttribute> get_attributes() { return mwnn_attributes; }

    std::vector<int> get_attribute_value(std::string name) {
      auto it = std::find_if(
      std::begin(mwnn_attributes), std::end(mwnn_attributes), [&](MWNNAttribute& attribute) {
          return attribute.get_name() == name;
      });
      if (it == std::end(mwnn_attributes)) {
          std::cout << "\n ERROR : End of Attributes!!! - Couldn't find " << name;
      }
      return it->get_data();
    }
    void update_attribute_value(std::string name, int value) {
      auto it = std::find_if(
      std::begin(mwnn_attributes), std::end(mwnn_attributes), [&](MWNNAttribute& attribute) {
          return attribute.get_name() == name;
      });
      if (it == std::end(mwnn_attributes)) {
          std::cout << "\n ERROR : End of Attributes!!! - Couldn't find " << name << " while updating its value!!!";
      }
      return it->set_data(value);
    }
    std::shared_ptr<op::Node> get_node() {
      if(op_type == "Conv") {
        return std::make_shared<op::Conv>(name, inputs, outputs,
                                          get_attribute_value("dilations"),
                                          get_attribute_value("strides"),
                                          get_attribute_value("pads"),
                                          get_attribute_value("activation")[0]);
      }
      else if(op_type == "DepthwiseConv") {
        return std::make_shared<op::DepthwiseConv>(name, inputs, outputs,
                                                   get_attribute_value("dilations"),
                                                   get_attribute_value("strides"),
                                                   get_attribute_value("pads"),
                                                   get_attribute_value("activation")[0]);
      }
      else if(op_type == "Relu") {
        return std::make_shared<op::Relu>(name, inputs, outputs);
      }
      else if(op_type == "Gemm") {
        return std::make_shared<op::Gemm>(name, inputs, outputs);
      }
      else if(op_type == "Clip") {
        return std::make_shared<op::Clip>(name, inputs, outputs);
      }
      else if(op_type == "Mul") {
        return std::make_shared<op::Mul>(name, inputs, outputs);
      }
      else if(op_type == "Add" || op_type == "Sum") {
        return std::make_shared<op::Add>(name, inputs, outputs);
      }
      else if(op_type == "GlobalAveragePool") {
        return std::make_shared<op::GlobalAvgPool>(name, inputs, outputs);
      }
      else if(op_type == "MaxPool") {
        return std::make_shared<op::MaxPool>(name, inputs, outputs);
      }
      else if(op_type == "AveragePool") {
        return std::make_shared<op::AvgPool>(name, inputs, outputs);
      }
      else if(op_type == "Concat") {
        return std::make_shared<op::Concat>(name, inputs, outputs);
      }
      else if(op_type == "LRN") {
        return std::make_shared<op::LRN>(name, inputs, outputs);
      }
      else if(op_type == "Flatten") {
        return std::make_shared<op::Flatten>(name, inputs, outputs);
      }
      else if(op_type == "Squeeze") {
        return std::make_shared<op::Squeeze>(name, inputs, outputs);
      }
      else if(op_type == "Unsqueeze") {
        return std::make_shared<op::Unsqueeze>(name, inputs, outputs);
      }
      else if(op_type == "Gather") {
        return std::make_shared<op::Gather>(name, inputs, outputs);
      }
      else if(op_type == "Shape") {
        return std::make_shared<op::Shape>(name, inputs, outputs);
      }
      else if(op_type == "BatchNormalization") {
        return std::make_shared<op::BatchNormalization>(name, inputs, outputs);
      }
      else if(op_type == "Split") {
        return std::make_shared<op::Split>(name, inputs, outputs);
      }
      else if(op_type == "Mean") {
        return std::make_shared<op::Mean>(name, inputs, outputs);
      }
      else if(op_type == "Pad") {
        return std::make_shared<op::Pad>(name, inputs, outputs);
      }
      else if(op_type == "StridedSlice") {
        return std::make_shared<op::StridedSlice>(name, inputs, outputs);
      }
      else if(op_type == "ChannelShuffle") {
        return std::make_shared<op::ChannelShuffle>(name, inputs, outputs);
      }
      else if(op_type == "FullyConnected") {
        return std::make_shared<op::FullyConnected>(name, inputs, outputs);
      }
      else if(op_type == "Reshape") {
        return std::make_shared<op::Reshape>(name, inputs, outputs);
      }
      else if(op_type == "Softmax") {
        return std::make_shared<op::Softmax>(name, inputs, outputs);
      }
      else
        return NULL;
    }
  private:
    std::string name;
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<MWNNAttribute> mwnn_attributes;
};

} //namespace metawarenn

#endif //METAWARENN_NODE_H_
