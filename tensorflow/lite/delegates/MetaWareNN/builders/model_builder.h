#ifndef TENSORFLOW_LITE_DELEGATES_METAWARENN_BUILDERS_MODEL_BUILDER_H_
#define TENSORFLOW_LITE_DELEGATES_METAWARENN_BUILDERS_MODEL_BUILDER_H_

#include <fcntl.h>
#include <unistd.h>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "metawarenn_lib/metawarenn_common.h"
#include "metawarenn_lib/metawarenn_graph.h"
#include "metawarenn_lib/metawarenn_tensor.h"
#include "metawarenn_lib/metawarenn_node.h"
#include "metawarenn_lib/metawarenn_attribute.h"
#include "metawarenn_lib/metawarenn_element.h"

#include "metawarenn_lib/optimizer/metawarenn_optimizer.h"
#include "metawarenn_lib/mwnnconvert/mwnn_protobuf/cpp_wrapper/MWNN.pb.h"
#include "metawarenn_lib/mwnnconvert/mwnn_to_proto.h"

#define INVOKE_NNAC 0

namespace tflite {
namespace delegates {
namespace metawarenn {

class IOpBuilder;

class ModelBuilder {
 public:
  ModelBuilder(std::vector<int> nodes);
  ~ModelBuilder() = default;
  void CreateMWNNNode(
      std::shared_ptr<::metawarenn::Graph> graph_ptr_,
      const std::string &node_name_,
      const std::string &node_op_type_,
      const std::vector<::metawarenn::Attribute> &node_attributes_,
      const std::vector<std::string> &node_inputs_,
      const std::vector<std::string> &node_outputs_);
  void CreateMWNNQuantParams(std::shared_ptr<::metawarenn::Graph> graph_ptr_,
                             TfLiteTensor tensor);
  std::shared_ptr<::metawarenn::Graph> BuildGraph(TfLiteContext* context,
                                                  std::string subgraph_name);
  TfLiteStatus MetaWareNNCompile(std::shared_ptr<::metawarenn::Graph> graph);

  template<class T>
  void ParseInput(TfLiteTensor input_tensor, 
                  std::shared_ptr<::metawarenn::Graph> graph_ptr);

  static ::metawarenn::Element::ElementType GetMWNNTypeTF(int tf_type) {
    switch (tf_type) {
      case kTfLiteBool: {
        return ::metawarenn::Element::ElementType::kBoolean;
      }
      case kTfLiteFloat64: {
          return ::metawarenn::Element::ElementType::kDouble;
      }
      case kTfLiteFloat16: {
          return ::metawarenn::Element::ElementType::kFloat16;
      }
      case kTfLiteFloat32: {
          return ::metawarenn::Element::ElementType::kFloat;
      }
      case kTfLiteInt8: {
          return ::metawarenn::Element::ElementType::kInt8;
      }
      case kTfLiteInt16: {
          return ::metawarenn::Element::ElementType::kInt16;
      }
      case kTfLiteInt32: {
          return ::metawarenn::Element::ElementType::kInt32;
      }
      case kTfLiteInt64: {
          return ::metawarenn::Element::ElementType::kInt64;
      }
      case kTfLiteUInt8: {
          return ::metawarenn::Element::ElementType::kUint8;
      }
      case kTfLiteNoType: {
          return ::metawarenn::Element::ElementType::kDynamic;
      }
      case kTfLiteString: {
          return ::metawarenn::Element::ElementType::kString;
      }
      case kTfLiteComplex64: {
          return ::metawarenn::Element::ElementType::kComplex64;
      }
      default: {
          return ::metawarenn::Element::ElementType::kDynamic;
      }
    }
  }

  private:
    std::vector<int> subgraph_nodes_;
};

} // namespace metaware
} // namespace delegates
} //namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_METAWARENN_BUILDERS_MODEL_BUILDER_H_
