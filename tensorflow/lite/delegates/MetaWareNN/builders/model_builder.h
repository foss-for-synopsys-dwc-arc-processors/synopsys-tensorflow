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
#include "metawarenn_lib/metawarenn_utils.h"
#include "metawarenn_lib/metawarenn_element.h"

#include "metawarenn_lib/optimizer/pass_manager.h"
#include "metawarenn_lib/optimizer/metawarenn_optimizer.h"
#include "metawarenn_lib/optimizer/remove_reshape.h"
#include "metawarenn_lib/optimizer/convert_layout.h"
#include "metawarenn_lib/optimizer/calculate_offset.h"

#include "metawarenn_lib/mwnnconvert/mwnn_protobuf/cpp_wrapper/MWNN.pb.h"
#include "metawarenn_lib/mwnnconvert/mwnn_to_proto.h"

#define HWC_TO_CHW 1
#define INVOKE_NNAC 0

namespace tflite {
namespace delegates {
namespace metawarenn {

class IOpBuilder;

class ModelBuilder {
 public:
  ModelBuilder(std::vector<int> nodes);
  ~ModelBuilder() = default;
  void CreateMWNNNode(std::shared_ptr<::metawarenn::Graph> graph_ptr_,
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
  void parse_input(TfLiteTensor input_tensor, std::shared_ptr<::metawarenn::Graph> graph_ptr);

  static ::metawarenn::ElementType::element_type get_mwnn_type_tf(int tf_type) {
      switch (tf_type) {
          case kTfLiteBool:
              return ::metawarenn::ElementType::element_type::boolean_;
          case kTfLiteFloat64:
              return ::metawarenn::ElementType::element_type::double_;
          case kTfLiteFloat16:
              return ::metawarenn::ElementType::element_type::float16_;
          case kTfLiteFloat32:
              return ::metawarenn::ElementType::element_type::float_;
          case kTfLiteInt8:
              return ::metawarenn::ElementType::element_type::int8_;
          case kTfLiteInt16:
              return ::metawarenn::ElementType::element_type::int16_;
          case kTfLiteInt32:
              return ::metawarenn::ElementType::element_type::int32_;
          case kTfLiteInt64:
              return ::metawarenn::ElementType::element_type::int64_;
          case kTfLiteUInt8:
              return ::metawarenn::ElementType::element_type::uint8_;
          case kTfLiteNoType:
              return ::metawarenn::ElementType::element_type::dynamic_;
          case kTfLiteString:
              return ::metawarenn::ElementType::element_type::string_;
          case kTfLiteComplex64:
              return ::metawarenn::ElementType::element_type::complex64_;
          default:
              return ::metawarenn::ElementType::element_type::dynamic_;
      }
  }


 private:
  std::vector<int> subgraph_nodes_;
};

} // namespace metaware
} // namespace delegates
} //namespace tflite
