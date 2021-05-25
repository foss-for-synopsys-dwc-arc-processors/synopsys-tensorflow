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
  return kTfLiteOk;
}

TfLiteStatus MetaWareNNDelegateKernel::Invoke(TfLiteContext* context,
                                           TfLiteNode* node) {
  std::cout<<"\nInside MetaWareNNDelegateKernel's Invoke!!!"<<std::endl;
  int is_HWC = HWC_TO_CHW ? 0 : 1;

  //Fills the graph_inputs with input data pointer using indexes
  std::unordered_map<std::string, float*> graph_inputs;
  std::unordered_map<std::string, float*> graph_outputs;

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
    }
  }

  /*namespace bip = boost::interprocess;
  bip::shared_memory_object shm(bip::open_only, "SharedMemoryFile", bip::read_only);
  bip::mapped_region region(shm, bip::read_only);
  bip::bufferstream bs(std::ios::in);
  bs.buffer(reinterpret_cast<char*>(region.get_address()), region.get_size());
  boost::archive::text_iarchive ia(bs);
  ::metawarenn::MWNNGraph mwnn_graph;
  ia >> mwnn_graph;
  convert_to_mwnn_format(mwnn_graph, is_HWC) ;
  bip::shared_memory_object::remove("SharedMemoryFile");*/

  std::cout << "\n In MWNN Kernel Invoke : " << mwnn_graph_->get_graph_nodes().size() << "  Graph Name : " << mwnn_graph_->get_name();
  convert_to_mwnn_format(*mwnn_graph_, graph_inputs, graph_outputs, is_HWC);

  return kTfLiteOk;
}
}  // namespace tflite
