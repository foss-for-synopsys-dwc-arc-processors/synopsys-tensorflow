#include <vector>
#include <iostream>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/delegates/MetaWareNN/MetaWareNN_delegate_kernel.h"

namespace tflite {

TfLiteStatus MetaWareNNDelegateKernel::Init(TfLiteContext* context,
                                         const TfLiteDelegateParams* params) {
  std::cout<<"\nInside MetaWareNNDelegateKernel's Init!!"<<std::endl;
  for (auto node_index : TfLiteIntArrayView(params->nodes_to_replace)) {
    nodes_.push_back(node_index);
  }
  model_builder_ = std::unique_ptr<delegates::metawarenn::ModelBuilder>
                   (new delegates::metawarenn::ModelBuilder(nodes_));
  mwnn_subgraph_counter_++;
  std::string subgraph_name = "MetaWareNN_" + std::to_string(mwnn_subgraph_counter_);
  mwnn_graph_ = model_builder_->BuildGraph(context, subgraph_name);
  return kTfLiteOk;
}

TfLiteStatus MetaWareNNDelegateKernel::Prepare(TfLiteContext* context,
                                           TfLiteNode* node) {
  std::cout<<"\nInside MetaWareNNDelegateKernel's Prepare!!"<<std::endl;
  if(model_builder_->MetaWareNNCompile(mwnn_graph_)) {
    graph_prepared_ = true;
  }
  std::cout << "\n In MWNN Kernel Prepare : " << mwnn_graph_->get_graph_nodes().size() << "  Graph Name : " << mwnn_graph_->get_name();
  mwnn_exe_graph_ = std::make_shared<metawarenn::MWNNExecutableGraph>(*mwnn_graph_);
  return kTfLiteOk;
}

TfLiteStatus MetaWareNNDelegateKernel::Invoke(TfLiteContext* context,
                                           TfLiteNode* node) {
  std::cout<<"\nInside MetaWareNNDelegateKernel's Invoke!!!"<<std::endl;
  int is_HWC = HWC_TO_CHW ? 0 : 1;

  //Fills the graph_inputs with input data pointer using indexes
  std::unordered_map<std::string, float*> graph_inputs;
  std::unordered_map<std::string, float*> graph_outputs;
  std::string output_tensor_name;

  for (int input_idx = 0; input_idx < node->inputs->size; ++input_idx) {
    const auto tensor_index = node->inputs->data[input_idx];
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    if (tensor->allocation_type == kTfLiteArenaRw && tensor->data.f != nullptr) { //Input - Data
      graph_inputs[tensor->name] = tensor->data.f;
    }
    else if(tensor->allocation_type == kTfLiteMmapRo && tensor->data.f != nullptr) { //Weights, Biases etc.,
      graph_inputs[tensor->name] = tensor->data.f;
    }
  }
  for (int output_idx = 0; output_idx < node->outputs->size; ++output_idx) {
    const auto tensor_index = node->outputs->data[output_idx];
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    if (tensor->allocation_type == kTfLiteArenaRw && tensor->data.f != nullptr) { //Output - Data
      graph_outputs[tensor->name] = tensor->data.f;
      output_tensor_name = tensor->name;
    }
  }

  std::cout << "\n In MWNN Kernel Invoke : " << mwnn_graph_->get_graph_nodes().size() << "  Graph Name : " << mwnn_graph_->get_name();

    // **************************************** Calls to invoke the MetaWareNN Inference API ************************************

    metawarenn::MWNNInferenceApi mwapi;

    std::vector<std::string> ip_names = mwnn_graph_->get_graph_ip_names();
    auto ip_shape = mwnn_graph_->get_graph_ip_tensor()[0].get_dims();

    mwapi.prepareInput(graph_inputs[ip_names[0]], ip_shape);
    auto op_shape = mwnn_graph_->get_graph_op_tensor()[0].get_dims();

    mwapi.prepareOutput(op_shape);

    mwapi.prepareGraph(mwnn_graph_->get_name());

    mwapi.runGraph();

    mwapi.getOutput(graph_outputs[output_tensor_name], op_shape);

    // ******************************************* Call to invoke the local run function *****************************************

    //convert_to_mwnn_format(*mwnn_graph_, graph_inputs, graph_outputs, is_HWC);
    //mwnn_exe_graph_->runGraph();

  return kTfLiteOk;
}
}  // namespace tflite
