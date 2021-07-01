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

#include <boost/serialization/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/streams/bufferstream.hpp>
#include <boost/serialization/vector.hpp>

#include <sys/ipc.h>
#include <sys/shm.h>
#include <fstream>

#define BUF_SIZE 7340032

struct shmseg {
   int cnt;
   char buf[BUF_SIZE];
};

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
  std::shared_ptr<::metawarenn::MWNNGraph> BuildGraph(TfLiteContext* context, std::string subgraph_name);
  TfLiteStatus MetaWareNNCompile(std::shared_ptr<::metawarenn::MWNNGraph> mwnn_graph);

 private:
  std::vector<int> subgraph_nodes_;
};

} // namespace metaware
} // namespace delegates
} //namespace tflite
