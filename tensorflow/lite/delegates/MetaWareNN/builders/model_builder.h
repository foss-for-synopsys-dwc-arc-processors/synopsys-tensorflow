#include "metawarenn_lib/metawarenn_model.h"
#include "metawarenn_lib/metawarenn_graph.h"
#include "metawarenn_lib/metawarenn_tensor.h"
#include "metawarenn_lib/metawarenn_node.h"
#include "metawarenn_lib/metawarenn_attribute.h"

#include "metawarenn_lib/optimizer/pass_manager.h"
#include "metawarenn_lib/optimizer/metawarenn_optimizer.h"
#include "metawarenn_lib/optimizer/dummy_pass_1.h"
#include "metawarenn_lib/optimizer/dummy_pass_2.h"
#include "metawarenn_lib/optimizer/dummy_pass_3.h"

namespace tflite {
namespace delegates {
namespace metawarenn {

class IOpBuilder;

class ModelBuilder {
 public:
  ModelBuilder(std::vector<int> nodes);
  ~ModelBuilder() = default;
  TfLiteStatus BuildGraph(TfLiteContext* context);
  TfLiteStatus MetaWareNNCompile();

 private:
  std::vector<int> subgraph_nodes_;
};

} // namespace metaware
} // namespace delegates
} //namespace tflite
