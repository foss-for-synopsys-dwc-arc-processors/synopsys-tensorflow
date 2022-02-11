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
  dynamic_shape_ = false;
  auto ip_tensor = graph_->get_graph_ip_tensor()[0];
  auto dims = ip_tensor.get_dims();
  auto name = ip_tensor.get_name();
  for(int i = 0; i < dims.size(); i++) {
    if(dims[i] == -1) {
      dynamic_shape_ = true;
      input_shape_range_[name][i] = std::make_pair(INT_MAX, INT_MIN);
    }
  }
  builder_config_ = inference_builder_->CreateBuilderConfig();

  inference_builder_->FillGraphDesc(graph_);

  // Create ExecutableGraph from MWNNGraph
  exe_graph_ = inference_builder_->CacheOrCreateExeGraph(graph_, graph_->get_name(), false);

  // dynamic_shape_ - yet to verify the flow
  if(!dynamic_shape_) {
    inference_engine_ = inference_builder_->CreateInferenceEngine(exe_graph_, builder_config_, false);
    inference_engine_->SerializeToFile();
    execution_context_ = inference_engine_->CreateExecutionContext();
  }
  #endif
  return kTfLiteOk;
}

TfLiteStatus MetaWareNNDelegateKernel::Invoke(TfLiteContext* context,
                                           TfLiteNode* node) {
  std::cout<<"\nInside MetaWareNNDelegateKernel's Invoke!!!"<<std::endl;
  int is_HWC = HWC_TO_CHW ? 0 : 1;

  #if INFERENCE_ENGINE

  bool update_engine = false;
  builder_config_ = inference_builder_->CreateBuilderConfig();
  if(dynamic_shape_) {
    bool profile_file_exists = false;
    //Creates a new optimization profile for dynamic input shapes
    if(optimization_profile_ == nullptr)
      optimization_profile_ = inference_builder_->CreateOptimizationProfile();
    auto profile_path = inference_builder_->GetProfilePath(graph_->get_name(), &profile_file_exists);
    if(profile_file_exists)
      inference_builder_->DeserializeProfileInfo(profile_path, builder_config_);
    builder_config_->PrintOptimizationProfileInfo();
  }

  //Fills the graph_inputs with input data pointer using indexes
  std::unordered_map<std::string, float*> graph_inputs;
  std::unordered_map<std::string, float*> graph_outputs;
  std::string output_tensor_name;

  for (int input_idx = 0; input_idx < node->inputs->size; ++input_idx) {
    const auto tensor_index = node->inputs->data[input_idx];
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    std::vector<int> tensor_shapes(tensor->dims->data, tensor->dims->data + tensor->dims->size);
    if (tensor->allocation_type == kTfLiteArenaRw && tensor->data.f != nullptr) { //Input - Data
      graph_inputs[tensor->name] = tensor->data.f;
    }
    else if(tensor->allocation_type == kTfLiteMmapRo && tensor->data.f != nullptr) { //Weights, Biases etc.,
      graph_inputs[tensor->name] = tensor->data.f;
    }
    //If graph input contains dynamic shape then get the size at runtime & fill the optimization profile attributes
    if(dynamic_shape_) {
      if(input_shape_range_.find(tensor->name) != input_shape_range_.end()) {
        auto& ip_shape_range_ = input_shape_range_[tensor->name];
        for(int d = 0; d < tensor_shapes.size(); d++) {
          if (ip_shape_range_.find(d) != ip_shape_range_.end()) {
            // Update Minimum Dimension
            if (tensor_shapes[d] < ip_shape_range_[d].first) {
              ip_shape_range_[d].first = tensor_shapes[d];
              update_engine = true;
            }
            // Update Maximum Dimension
            if(tensor_shapes[d] > ip_shape_range_[d].second) {
              ip_shape_range_[d].second = tensor_shapes[d];
              update_engine = true;
            }
          }
        }
        optimization_profile_->SetInputDimensions(tensor->name, ip_shape_range_);
      }
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

  if (dynamic_shape_) {
    std::cout << "\n Creating Engine, Context for Dynamic Input shapes";
    builder_config_->AddOptimizationProfile(optimization_profile_);
    inference_engine_ = inference_builder_->CreateInferenceEngine(exe_graph_, builder_config_, update_engine);
    auto graph_desc = inference_engine_->GetGraphDesc();
    const auto tensor_index = node->inputs->data[0];
    TfLiteTensor* tensor = &context->tensors[tensor_index];
    std::vector<int> tensor_shapes(tensor->dims->data, tensor->dims->data + tensor->dims->size);
    uint64_t size = 1;
    std::cout << "\n ORT Input Shape: ";
    for(auto dim: tensor_shapes) {
      std::cout << dim << ", ";
      size = size * dim;
    }

    graph_desc.UpdateInputDesc(0, size * sizeof(::metawarenn::data_type));
    inference_engine_->SetGraphDesc(graph_desc);

    inference_engine_->SerializeToFile();
    execution_context_ = inference_engine_->CreateExecutionContext();
  }

  std::cout << "\n In MWNN Kernel Invoke : " << graph_->get_graph_nodes().size() << "  Graph Name : " << graph_->get_name();

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