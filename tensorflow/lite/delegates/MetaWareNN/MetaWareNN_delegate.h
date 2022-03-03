#ifndef TENSORFLOW_LITE_DELEGATES_METAWARENN_METAWARENN_DELEGATE_H_
#define TENSORFLOW_LITE_DELEGATES_METAWARENN_METAWARENN_DELEGATE_H_

#include "tensorflow/lite/c/common.h"
#include <memory>
#include "tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/mwnnconvert/json/include/json.hpp"
using json = nlohmann::json;

struct TfLiteMetaWareNNDelegateOptions {
  int32_t max_delegated_partitions;
  int min_nodes_per_partition;
};

TfLiteDelegate* TfLiteMetaWareNNDelegateCreate(
    const TfLiteMetaWareNNDelegateOptions* options);
void TfLiteMetaWareNNDelegateDelete(TfLiteDelegate* delegate);

#endif // TENSORFLOW_LITE_DELEGATES_METAWARENN_METAWARENN_DELEGATE_H_
