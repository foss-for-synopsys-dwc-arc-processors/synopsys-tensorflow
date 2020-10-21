#ifndef TENSORFLOW_LITE_DELEGATES_METAWARENN_METAWARENN_DELEGATE_KERNEL_H_
#define TENSORFLOW_LITE_DELEGATES_METAWARENN_METAWARENN_DELEGATE_KERNEL_H_

#include <time.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/delegates/MetaWareNN/builders/model_builder.h"

namespace tflite {

class MetaWareNNDelegateKernel {
 public:

  TfLiteStatus Init(TfLiteContext* context, const TfLiteDelegateParams* params);

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node);

  TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node);


 private:

  const MetaWareNN* metawarenn_ = nullptr;

  //Create model builder
  std::unique_ptr<delegates::metawarenn::ModelBuilder> model_builder_;

  // Indices of nodes in the delegated TfLite subgraph.
  std::vector<int> nodes_;

  // Whether the MetawareNN graph is prepared or not.
  bool graph_prepared_ = false;
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_METAWARENN_METAWARENN_DELEGATE_KERNEL_H_
