#ifndef TENSORFLOW_LITE_DELEGATES_METAWARENN_METAWARENN_DELEGATE_KERNEL_H_
#define TENSORFLOW_LITE_DELEGATES_METAWARENN_METAWARENN_DELEGATE_KERNEL_H_

#include <time.h>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/MetaWareNN/builders/model_builder.h"
#include "tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/mwnnconvert/mwnn_to_onnx_proto.h"
#include "tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/inference_engine/mwnn_inference_engine.h"
#include "tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/inference_engine/mwnn_builder.h"

namespace tflite {

static int kSubgraphCounter = 0;

class MetaWareNNDelegateKernel {
 public:
  TfLiteStatus Init(TfLiteContext* context, const TfLiteDelegateParams* params);
  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node);
  TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node);
 private:
  std::unique_ptr<delegates::metawarenn::ModelBuilder> model_builder_;
  std::shared_ptr<::metawarenn::Graph> graph_;
  // Indices of nodes in the delegated TfLite subgraph.
  std::vector<int> nodes_;

  #if INFERENCE_ENGINE
  std::shared_ptr<metawarenn::OptimizationProfile> 
      optimization_profile_ = nullptr;
  std::shared_ptr<metawarenn::BuilderConfig> builder_config_;
  std::shared_ptr<metawarenn::Builder> 
      inference_builder_ = std::make_shared<metawarenn::Builder>();
  std::shared_ptr<metawarenn::ExecutableGraph> exe_graph_;
  std::shared_ptr<metawarenn::InferenceEngine> inference_engine_;
  std::shared_ptr<metawarenn::ExecutionContext> execution_context_;
  std::unordered_map<std::string, std::unordered_map<size_t, 
                     std::pair<int64_t, int64_t>>> input_shape_range_;
  bool dynamic_shape_;
  #endif
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_METAWARENN_METAWARENN_DELEGATE_KERNEL_H_
