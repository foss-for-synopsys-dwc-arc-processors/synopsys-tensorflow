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
  subgraph_counter_++;
  std::string subgraph_name = "MetaWareNN_" + std::to_string(subgraph_counter_);
  graph_ = model_builder_->BuildGraph(context, subgraph_name);
  return kTfLiteOk;
}

TfLiteStatus MetaWareNNDelegateKernel::Prepare(TfLiteContext* context,
                                           TfLiteNode* node) {
  std::cout<<"\nInside MetaWareNNDelegateKernel's Prepare!!"<<std::endl;
  if(model_builder_->MetaWareNNCompile(graph_)) {
    graph_prepared_ = true;
  }
  std::cout << "\n In MWNN Kernel Prepare : " << graph_->get_graph_nodes().size() << "  Graph Name : " << graph_->get_name();
  #if !EXECUTABLE_GRAPH_SERIALIZATION
  write_onnx_proto(graph_);
  #endif

  #if INFERENCE_ENGINE
  inference_engine_ = inference_builder_->CreateInferenceEngine(*graph_);
  inference_engine_->SerializeToFile();
  execution_context_ = inference_engine_->CreateExecutionContext();
  #endif
  /*#if EXECUTABLE_GRAPH_SERIALIZATION
  exe_graph_ = std::make_shared<metawarenn::ExecutableGraph>(*graph_);
  #endif*/
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

  std::cout << "\n In MWNN Kernel Invoke : " << graph_->get_graph_nodes().size() << "  Graph Name : " << graph_->get_name();

  #if INFERENCE_ENGINE
  auto graph_desc = inference_engine_->GetGraphDesc();
  std::string ip_name = graph_desc.input_desc[0].tensor_name;
  std::string op_name = graph_desc.output_desc[0].tensor_name;
  std::cout << "\n Ip_name : " << ip_name << "\t Size : " << graph_desc.input_desc[0].size;
  std::cout << "\n Op_name : " << op_name << "\t Size : " << graph_desc.output_desc[0].size;

  execution_context_->CopyInputToDevice(graph_inputs[ip_name], graph_desc.input_desc[0].size);
  execution_context_->Execute();
  execution_context_->CopyOutputFromDevice(graph_outputs[op_name], graph_desc.output_desc[0].size);
  #endif

    // ******************************************* Call to invoke the local run function *****************************************

    //convert_to_mwnn_format(*graph_, graph_inputs, graph_outputs, is_HWC);

  return kTfLiteOk;
}
}  // namespace tflite