#include <iostream>

#include "model_builder.h"
#include "op_builder.h"
#include "helper.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/MetaWareNN/MetaWareNN_lib/MetaWareNN_implementation.h" //To access MetawareNN APIs from shared library

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
  /* Create and Populate the metawarenn_model_ by adding ops and operands using MetaWareNN API */
  TF_LITE_ENSURE_STATUS(AddOperations(context));
  return kTfLiteOk;
}

TfLiteStatus ModelBuilder::MetaWareNNCompile() {
  return kTfLiteOk;
}

IOpBuilder* ModelBuilder::GetOpBuilder(int32_t op_type) {
  if (!Contains(op_builders_, op_type))
    return nullptr;
  return op_builders_[op_type].get();
}

TfLiteStatus ModelBuilder::AddOperations(TfLiteContext* context) {
  std::cout<<"\nAddOperations!!\n"<<std::endl;
  for (size_t node_index = 0; node_index < subgraph_nodes_.size(); node_index++) {
    TfLiteNode* node;
    TfLiteRegistration* reg;
    const auto status = context->GetNodeAndRegistration(context, node_index,
                                                         &node, &reg);
    auto op_type = reg->builtin_code;
    if (auto* op_builder = GetOpBuilder(op_type)) {
      TF_LITE_ENSURE_STATUS(op_builder->AddToModelBuilder(*this, op_type));
    }
  }
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
