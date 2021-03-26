#ifndef METAWARENN_COMMON_H_
#define METAWARENN_COMMON_H_

#define GLOW 0

//ONNXRuntime
#include "onnx/onnx-ml.pb.h"

//TFLite
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <map>
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/schema/schema_generated.h"

//GLOW
#if GLOW
#include "Glow/Graph/Graph.h"
#include "Glow/Graph/Utils.h"
#endif

#include <boost/serialization/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/streams/bufferstream.hpp>
#include <boost/serialization/vector.hpp>

using namespace onnx;
#if GLOW
using namespace glow;
#endif

#endif //METAWARENN_COMMON_H_
