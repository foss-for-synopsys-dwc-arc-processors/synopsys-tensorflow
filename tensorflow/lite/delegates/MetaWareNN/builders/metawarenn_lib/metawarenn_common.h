#ifndef METAWARENN_COMMON_H_
#define METAWARENN_COMMON_H_

#define ONNX 0
#define TFLITE 1
#define GLOW 0
#define TVM 0

//Common headers
#include <sys/stat.h>
#include <sys/types.h>

//ONNXRuntime
#if ONNX
#include "onnx/onnx-ml.pb.h"
#include <numeric>
#endif

//TFLite
#if TFLITE
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <map>
#include <fcntl.h>
#include <unordered_map>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/schema/schema_generated.h"
#endif

//GLOW
#if GLOW
#include "Glow/Graph/Graph.h"
#include "Glow/Graph/Utils.h"
#include "glow/Backend/BackendUtils.h"
#endif

//TVM
#if TVM
#include <numeric>
#include <regex>
#include "tvm/json/json_node.h"
#include "tvm/json/json_runtime.h"
#endif

#if ONNX
using namespace onnx;
#endif
#if GLOW
using namespace glow;
#endif
#if TVM
using namespace tvm::runtime::json;
#endif
#endif //METAWARENN_COMMON_H_
