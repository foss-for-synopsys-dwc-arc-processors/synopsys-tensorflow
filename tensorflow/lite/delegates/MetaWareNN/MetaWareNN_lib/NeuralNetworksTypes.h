#ifndef TENSORFLOW_LITE_NNAPI_NEURALNETWORKSTYPES_H_
#define TENSORFLOW_LITE_NNAPI_NEURALNETWORKSTYPES_H_

#include <stdint.h>
#include <stdio.h>
#include <string>

typedef struct MetaWareNNModel MetaWareNNModel;

typedef struct MetaWareNNCompilation MetaWareNNCompilation;

typedef struct MetaWareNNExecution MetaWareNNExecution;

/**
 * Operation types.
 *
 * The type of operations that can be added to a model.
 */
enum {
  METAWARENN_ADD = 0,
  METAWARENN_GLOBAL_AVERAGE_POOL_2D = 1,
  METAWARENN_CONV_2D = 3,
  METAWARENN_DEPTHWISECONV_2D = 4,
  METAWARENN_RELU = 4,
  METAWARENN_RESHAPE = 5,
  METAWARENN_SOFTMAX = 6
};

#endif  // TENSORFLOW_LITE_NNAPI_NEURALNETWORKSTYPES_H_
