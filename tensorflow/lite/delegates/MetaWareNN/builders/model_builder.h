#include "metawarenn_lib/metawarenn_common.h"
#include "metawarenn_lib/metawarenn_graph.h"
#include "metawarenn_lib/metawarenn_tensor.h"
#include "metawarenn_lib/metawarenn_node.h"
#include "metawarenn_lib/metawarenn_attribute.h"
#include "metawarenn_lib/metawarenn_utils.h"

#include "metawarenn_lib/optimizer/pass_manager.h"
#include "metawarenn_lib/optimizer/metawarenn_optimizer.h"
#include "metawarenn_lib/optimizer/remove_reshape.h"
#include "metawarenn_lib/optimizer/convert_layout.h"

#include "metawarenn_lib/mwnnconvert/mwnn_protobuf/cpp_wrapper/MWNN.pb.h"

#define HWC_TO_CHW 1
#define INVOKE_NNAC 1

namespace tflite {
namespace delegates {
namespace metawarenn {

class IOpBuilder;

class ModelBuilder {
 public:
  ModelBuilder(std::vector<int> nodes);
  ~ModelBuilder() = default;
  ::metawarenn::MWNNGraph BuildGraph(TfLiteContext* context);
  TfLiteStatus MetaWareNNCompile(::metawarenn::MWNNGraph *mwnn_graph);

 private:
  std::vector<int> subgraph_nodes_;
};

} // namespace metaware
} // namespace delegates
} //namespace tflite
