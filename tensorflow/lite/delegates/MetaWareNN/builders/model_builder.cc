#include <iostream>

#include "model_builder.h"
#include "op_builder.h"
#include "helper.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/MetaWareNN/MetaWareNN_lib/MetaWareNN_implementation.h" //To access MetawareNN APIs from shared library
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"

namespace tflite {
namespace delegates {
namespace metawarenn{

ModelBuilder::ModelBuilder(std::vector<int> nodes)
    : metawarenn_(MetaWareNNImplementation()), subgraph_nodes_(nodes) {
  op_builders_ = CreateOpBuilders();
}

TfLiteStatus ModelBuilder::BuildGraph(TfLiteContext* context) {
  std::cout<<"\nBuildGraph!!"<<std::endl;
  metawarenn_model_ = std::unique_ptr<delegates::metawarenn::Model>
                      (new delegates::metawarenn::Model());
  mwnn_graph_.set_name("MetaWareNN_NodeSubSet_1");
  /* Create and Populate the metawarenn_model_ by adding ops and operands using MetaWareNN API */
  TF_LITE_ENSURE_STATUS(AddOperations(context));
  return kTfLiteOk;
}

/* TODO: High Level Graph to MetaWareNN Graph Representation,
         Apply Passes on MetaWareNN Graph,
         Generate Low Level Graph to run on devices*/
TfLiteStatus ModelBuilder::MetaWareNNCompile() {
  std::cout << "\n In MetaWareNNCompile !!! ";
  //Call Passes
  ::metawarenn::optimizer::PassManager manager;
  ::metawarenn::optimizer::DummyPass1 d_pass1(7);
  std::cout << "\n MetaWareNNCC : " << d_pass1.get_name();
  ::metawarenn::optimizer::DummyPass2 d_pass2;
  std::cout << "\n MetaWareNNCC : " << d_pass2.get_name();
  ::metawarenn::optimizer::DummyPass3 d_pass3;
  std::cout << "\n MetaWareNNCC : " << d_pass3.get_name();
  ::metawarenn::optimizer::DummyPass1 d_pass4;
  std::cout << "\n MetaWareNNCC : " << d_pass4.get_name();
  manager.register_pass(d_pass1);
  manager.register_pass(d_pass2);
  manager.register_pass(d_pass3);
  manager.register_pass(d_pass4);
  manager.run_passes();
  return kTfLiteOk;
}

IOpBuilder* ModelBuilder::GetOpBuilder(int32_t op_type) {
  if (!Contains(op_builders_, op_type))
    return nullptr;
  return op_builders_[op_type].get();
}

TfLiteStatus ModelBuilder::AddOperations(TfLiteContext* context) {
  std::cout<<"\nAddOperations!!\n"<<std::endl;
  std::cout << "\n----------------------------------------------------------------------------------------------------------------\n";
  std::cout << "\n MWNN Graph Name : " << mwnn_graph_.get_name();
  for (size_t node_index = 0; node_index < subgraph_nodes_.size(); node_index++) {
    TfLiteNode* node;
    TfLiteRegistration* reg;
    const auto status = context->GetNodeAndRegistration(context, node_index,
                                                         &node, &reg);
    auto op_type = reg->builtin_code;

    std::string node_name;
    std::string node_op_type;
    std::vector<std::string> node_inputs;
    std::vector<std::string> node_outputs;

    //Op Names are added to follow the same pattern like in ONNX as of now.
    if (op_type == kTfLiteBuiltinConv2d) {
      node_op_type = "Conv";
      node_name = node_op_type + std::to_string(node_index);
    }
    else if (op_type == kTfLiteBuiltinDepthwiseConv2d) {
      node_op_type = "Conv";
      node_name = node_op_type + std::to_string(node_index);
    }
    else if (op_type == kTfLiteBuiltinAveragePool2d) {
      node_op_type = "GlobalAveragePool";
      node_name = node_op_type + std::to_string(node_index);
    }
    else if (op_type == kTfLiteBuiltinAdd) {
      node_op_type = "Add";
      node_name = node_op_type + std::to_string(node_index);
    }
    else if (op_type == kTfLiteBuiltinRelu) {
      node_op_type = "Relu";
      node_name = node_op_type + std::to_string(node_index);
    }
    else if (op_type == kTfLiteBuiltinReshape) {
      node_op_type = "Reshape";
      node_name = node_op_type + std::to_string(node_index);
    }
    else if (op_type == kTfLiteBuiltinSoftmax) {
      node_op_type = "Softmax";
      node_name = node_op_type + std::to_string(node_index);
    }
    else {
      std::cout << "\n Unsupported op_type: " << op_type;
      exit(1);
    }

    for (int i = 0; i < node->inputs->size; ++i) {
      const int tensor_id = node->inputs->data[i];
      node_inputs.emplace_back(context->tensors[tensor_id].name);
    }

    for (int i = 0; i < node->outputs->size; ++i) {
      const int tensor_id = node->outputs->data[i];
      node_outputs.emplace_back(context->tensors[tensor_id].name);
    }

    ::metawarenn::MWNNNode mwnn_node(node_name, node_op_type, node_inputs, node_outputs);
    mwnn_graph_.set_graph_nodes(mwnn_node);

    for (int i = 0; i < node->inputs->size; ++i) {
      const int tensor_id = node->inputs->data[i];
      const auto& input_tensor = context->tensors[tensor_id];

      if (input_tensor.allocation_type == kTfLiteMmapRo) {
          std::vector<int> dims_vec(input_tensor.dims->data, input_tensor.dims->data + input_tensor.dims->size);
          auto num_tensor_elements = std::accumulate(begin(dims_vec), end(dims_vec), 1, std::multiplies<int>());
          std::vector<float> tensor_vec(input_tensor.data.f, input_tensor.data.f + num_tensor_elements);
          ::metawarenn::MWNNTensor mwnn_tensor(input_tensor.name, dims_vec,  tensor_vec);
          mwnn_graph_.set_graph_initializers(mwnn_tensor);
      }
    }

    if (auto* op_builder = GetOpBuilder(op_type)) {
      TF_LITE_ENSURE_STATUS(op_builder->AddToModelBuilder(*this, op_type));
    }
  }
  std::cout << "\n----------------------------------------------------------------------------------------------------------------\n";
  /*for (auto& it : mwnn_graph_.get_graph_initializers()) {
    std::cout << "\n Tensor Name: " << it.get_name();
  }
  for (auto& it : mwnn_graph_.get_graph_nodes()) {
    std::cout << "\n Node Name: " << it.get_name();
  }*/
  return kTfLiteOk;
}

TfLiteStatus ModelBuilder::AddOperation(int op) {
  if(op <0 && op>6){
    std::cout<<"Invalid op";
  }
  //std::cout<<"\nAddOperation for OP code - "<<op<<std::endl;
  return kTfLiteOk;
}

} // namespace metawarenn
} // namespace delegates
} // namespace tflite
