#include "fuse_relu.h"

namespace metawarenn {

namespace optimizer {

FuseRelu::FuseRelu() {
  set_name("FuseRelu");
}
FuseRelu::FuseRelu(std::shared_ptr<MWNNGraph> mwnn_graph, MWNNNode mwnn_node) {
  set_name("FuseRelu");
  graph = mwnn_graph;
  node = mwnn_node;
}
void FuseRelu::RunPass() {
  for (auto g_n : graph->get_graph_nodes()) {
    //To get consumer of current op
    for(int i = 0; i < (g_n.get_inputs()).size(); i++) {
      if(node.get_outputs()[0] == g_n.get_inputs()[i]) {
        consumers.insert(g_n.get_name());
      }
    }
    //To get producer of current op
    for(int i = 0; i < (g_n.get_outputs()).size(); i++) {
      if(node.get_inputs()[0] == g_n.get_outputs()[i]) {
        producers.insert(g_n.get_name());
        if(g_n.get_op_type() == "Conv" or g_n.get_op_type() == "DepthwiseConv") {
          graph->update_node_outputs(g_n.get_name(), node.get_outputs()[0], i);
          graph->update_node_attribute(g_n.get_name(), "activation", 1);
          graph->remove_nodes(node.get_name());
          graph->remove_graph_nodes(node.get_name());
          break;
        }
      }
    }
  }
  /*for (auto itr = consumers.begin(); itr != consumers.end(); ++itr) {
    std::cout << "\nConsumers : " << *itr;
  }
  for (auto itr = producers.begin(); itr != producers.end(); ++itr) {
    std::cout << "\nProducers : " << *itr;
  }*/
  }
} //namespace optimizer

} //namespace metawarenn
