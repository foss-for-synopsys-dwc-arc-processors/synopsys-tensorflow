#include "model.h"
#include <unordered_map>
#include <memory>
#include <numeric>
#include "tensorflow/lite/c/common.h"

#include "metawarenn_lib/metawarenn_model.h"
#include "metawarenn_lib/metawarenn_graph.h"
#include "metawarenn_lib/metawarenn_tensor.h"
#include "metawarenn_lib/metawarenn_node.h"

namespace tflite {
namespace delegates {
namespace metawarenn {

class IOpBuilder;

class ModelBuilder {
 public:
  ModelBuilder(std::vector<int> nodes);
  ~ModelBuilder() = default;

  TfLiteStatus AddOperation(int op);

  TfLiteStatus BuildGraph(TfLiteContext* context);

  TfLiteStatus MetaWareNNCompile();

 private:
  const MetaWareNN* metawarenn_{nullptr};
  std::vector<int> subgraph_nodes_;
  std::unique_ptr<Model> metawarenn_model_;
  std::unordered_map<std::int32_t, std::shared_ptr<IOpBuilder>> op_builders_;

  TfLiteStatus AddOperations(TfLiteContext* context) ;

  IOpBuilder* GetOpBuilder(int32_t op_type);

  ::metawarenn::MWNNModel mwnn_model_;
  ::metawarenn::MWNNGraph mwnn_graph_;
};

} // namespace metaware
} // namespace delegates
} //namespace tflite
