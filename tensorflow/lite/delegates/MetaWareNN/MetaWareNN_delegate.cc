#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/utils.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/delegates/MetaWareNN/MetaWareNN_delegate.h"
#include "tensorflow/lite/delegates/MetaWareNN/MetaWareNN_delegate_kernel.h"

namespace tflite {
namespace {

TfLiteRegistration GetMetaWareNNKernelRegistration() {
  std::cout<<"\nInside GetMetaWareNNKernelRegistration"<<std::endl;
  TfLiteRegistration kernel_registration;
  kernel_registration.profiling_string = nullptr;
  kernel_registration.builtin_code = kTfLiteBuiltinDelegate;
  kernel_registration.custom_name = "TfLiteMetaWareNNDelegate";
  kernel_registration.free = [](TfLiteContext* context, void* buffer) -> void {
    delete reinterpret_cast<MetaWareNNDelegateKernel*>(buffer);
  };

  kernel_registration.init = [](TfLiteContext* context, const char* buffer,
                                size_t length) -> void* {
    const TfLiteDelegateParams* params =
        reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    auto metawarenn_kernel = std::make_unique<MetaWareNNDelegateKernel>();
    if (metawarenn_kernel->Init(context, params) != kTfLiteOk) {
      return nullptr;
    }
    return metawarenn_kernel.release();
  };

  kernel_registration.invoke = [](TfLiteContext* context,
                                  TfLiteNode* node) -> TfLiteStatus {
    MetaWareNNDelegateKernel* kernel =
        reinterpret_cast<MetaWareNNDelegateKernel*>(node->user_data);
    if (!kernel) {
      context->ReportError(context, "MetaWareNN Kernel was not initialized");
      return kTfLiteError;
    }
    return kernel->Invoke(context, node);
  };
  kernel_registration.prepare = [](TfLiteContext* context,
                                   TfLiteNode* node) -> TfLiteStatus {
    if (node->user_data == nullptr) {
      context->ReportError(context, "MetaWareNN Kernel was not initialized");
      return kTfLiteError;
    }
    MetaWareNNDelegateKernel* kernel =
        reinterpret_cast<MetaWareNNDelegateKernel*>(node->user_data);
    return kernel->Prepare(context, node);
  };

  return kernel_registration;
}

class MetaWareNNDelegate: public TfLiteDelegate {
 public:
  explicit MetaWareNNDelegate(const TfLiteMetaWareNNDelegateOptions* params)
      : params_(params != nullptr ? *params
                                  : TfLiteMetaWareNNDelegateOptions({0})) {}

  TfLiteMetaWareNNDelegateOptions* params() { return &params_; }

  ~MetaWareNNDelegate() {
  }

 private:
  TfLiteMetaWareNNDelegateOptions params_;
};

bool IsNodeSupportedByMetaWareNN(const TfLiteRegistration* registration,
                              const TfLiteNode* node, TfLiteContext* context){
  switch (registration->builtin_code) {
    case kTfLiteBuiltinAdd:
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinDepthwiseConv2d:
    case kTfLiteBuiltinAveragePool2d:
    case kTfLiteBuiltinMaxPool2d:
    case kTfLiteBuiltinMul:
    case kTfLiteBuiltinConcatenation:
    case kTfLiteBuiltinFullyConnected:
    case kTfLiteBuiltinMean:
    case kTfLiteBuiltinSplit:
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinStridedSlice:
    case kTfLiteBuiltinSqueeze:
    case kTfLiteBuiltinRelu:
    case kTfLiteBuiltinReshape:
    case kTfLiteBuiltinSoftmax:
      return true;
    case kTfLiteBuiltinHardSwish:
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinSpaceToDepth:
    case kTfLiteBuiltinArgMax:
    case kTfLiteBuiltinTransposeConv:
    case kTfLiteBuiltinLogistic:
    case kTfLiteBuiltinSum:
    case kTfLiteBuiltinResizeNearestNeighbor:
    case kTfLiteBuiltinResizeBilinear:
    case kTfLiteBuiltinPrelu:
    case kTfLiteBuiltinSpaceToBatchNd:
      return true;
    case kTfLiteBuiltinDequantize:
      std::cout<< "\nWarning in MetaWareNN_delegate.cc: currently only support dequantizing float16->float32\n";
      return true;
    default:
      std::cout<< "\nMetaWareNN unsupported node enum: " << registration->builtin_code;
      std::cout<< "\nunsupported in MetaWareNN_delegate.cc\n"<<std::flush;
      exit(-4);
      return false;
  }
  return false;
}

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  std::cout<<"\nIn DelegatePrepare"<<std::endl;
  delegates::IsNodeSupportedFn node_supported_fn =
      [=](TfLiteContext* context, TfLiteNode* node,
          TfLiteRegistration* registration,
          std::string* unsupported_details) -> bool {
    return IsNodeSupportedByMetaWareNN(registration, node, context);
  };
  delegates::GraphPartitionHelper helper(context, node_supported_fn);
  TF_LITE_ENSURE_STATUS(helper.Partition(nullptr));

  TfLiteMetaWareNNDelegateOptions* params =
      static_cast<TfLiteMetaWareNNDelegateOptions*>(delegate->data_);

  if (params->max_delegated_partitions <= 0)
    params->max_delegated_partitions = std::numeric_limits<int>::max();

  std::vector<int> supported_nodes = helper.GetNodesOfFirstNLargestPartitions(
      params->max_delegated_partitions, params->min_nodes_per_partition);

  auto* metawarenn_delegate = static_cast<MetaWareNNDelegate*>(delegate);

  return context->ReplaceNodeSubsetsWithDelegateKernels(
      context, GetMetaWareNNKernelRegistration(),
      BuildTfLiteIntArray(supported_nodes).get(), delegate);
}

TfLiteDelegate* CreateDelegate(const TfLiteMetaWareNNDelegateOptions* params) {
  TfLiteDelegate* delegate = new MetaWareNNDelegate(params);
  std::cout<<"\nInside CreateDelegate"<<std::endl;
  delegate->data_ = static_cast<MetaWareNNDelegate*>(delegate)->params();
  delegate->flags = kTfLiteDelegateFlagsAllowDynamicTensors;
  delegate->Prepare = &DelegatePrepare;
  delegate->CopyFromBufferHandle = nullptr;
  delegate->CopyToBufferHandle = nullptr;
  delegate->FreeBufferHandle = nullptr;
  return delegate;
}

}  // namespace
}  // namespace tflite

TfLiteDelegate* TfLiteMetaWareNNDelegateCreate(
    const TfLiteMetaWareNNDelegateOptions* options) {
  std::cout<<"\nInside TfLiteMetaWareNNDelegateCreate!!"<<std::endl;
  return tflite::CreateDelegate(options);
}

void TfLiteMetaWareNNDelegateDelete(TfLiteDelegate* delegate) {
  delete delegate;
}
