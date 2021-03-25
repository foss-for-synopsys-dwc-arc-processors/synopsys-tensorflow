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
  mwnn_graph_ = model_builder_->BuildGraph(context);
  return kTfLiteOk;
}

TfLiteStatus MetaWareNNDelegateKernel::Prepare(TfLiteContext* context,
                                           TfLiteNode* node) {
  std::cout<<"\nInside MetaWareNNDelegateKernel's Prepare!!"<<std::endl;

  if(model_builder_->MetaWareNNCompile(&mwnn_graph_)) {
    graph_prepared_ = true;
  }
  return kTfLiteOk;
}

TfLiteStatus MetaWareNNDelegateKernel::Invoke(TfLiteContext* context,
                                           TfLiteNode* node) {
  std::cout<<"\nInside MetaWareNNDelegateKernel's Invoke!!!"<<std::endl;
  int is_HWC = HWC_TO_CHW ? 0 : 1;
  namespace bip = boost::interprocess;
  bip::shared_memory_object shm(bip::open_only, "SharedMemoryFile", bip::read_only);
  bip::mapped_region region(shm, bip::read_only);
  bip::bufferstream bs(std::ios::in);
  bs.buffer(reinterpret_cast<char*>(region.get_address()), region.get_size());
  boost::archive::text_iarchive ia(bs);
  ::metawarenn::MWNNGraph mwnn_graph;
  ia >> mwnn_graph;
  convert_to_mwnn_format(mwnn_graph, is_HWC) ;
  bip::shared_memory_object::remove("SharedMemoryFile");

  return kTfLiteOk;
}
}  // namespace tflite
