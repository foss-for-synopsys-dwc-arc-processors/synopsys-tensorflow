#ifndef METAWARENN_GRAPH_H_
#define METAWARENN_GRAPH_H_

#include "metawarenn_model.h"
#include "metawarenn_tensor.h"
#include "metawarenn_node.h"
#include "metawarenn_value_info.h"
#include "op/node.h"

namespace metawarenn {

class MWNNGraph {
  public:
    MWNNGraph() = default;
    MWNNGraph(TfLiteContext* context, std::vector<int> subgraph_nodes_);
    std::string get_name() { return name; }
    std::string get_graph_ip_name() { return ip_name; }
    std::string get_graph_op_name() { return op_name; }
    std::vector<MWNNTensor> get_graph_initializers() { return mwnn_initializer_tensors; }
    MWNNTensor get_initializer_tensor(std::string name) {
      auto it = std::find_if(
      std::begin(mwnn_initializer_tensors), std::end(mwnn_initializer_tensors), [&](MWNNTensor& tensor) {
          return tensor.get_name() == name;
      });
      if (it == std::end(mwnn_initializer_tensors)) {
          std::cout << "\n ERROR : End of Initializers!!! - Couldn't find " << name;
      }
      return *it;
    }
    std::vector<MWNNNode> get_graph_nodes() { return mwnn_nodes; }
    std::vector<MWNNValueInfo> get_graph_inputs() { return mwnn_inputs; }
    std::vector<MWNNValueInfo> get_graph_outputs() { return mwnn_outputs; }

    std::set<std::string> mwnn_initializer_names;
    std::map<std::string, op::Node> mwnn_graph_nodes;
  private:
    MWNNModel mwnn_model;
    std::string name;
    std::string ip_name;
    std::string op_name;
    std::vector<MWNNTensor> mwnn_initializer_tensors;
    std::vector<MWNNNode> mwnn_nodes;
    std::vector<MWNNValueInfo> mwnn_inputs;
    std::vector<MWNNValueInfo> mwnn_outputs;
};

} //namespace metawarenn
#endif //METAWARENN_GRAPH_H_
