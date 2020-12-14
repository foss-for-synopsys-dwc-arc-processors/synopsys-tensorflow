#ifndef METAWARENN_GRAPH_H_
#define METAWARENN_GRAPH_H_

#include "metawarenn_model.h"
#include "metawarenn_tensor.h"
#include "metawarenn_node.h"

namespace metawarenn {

class MWNNGraph {
  public:
     MWNNGraph() = default;
    void set_name(std::string m_name) { name = m_name; }
    std::string get_name() { return name; }
    std::vector<MWNNTensor> get_graph_initializers() { return mwnn_initializer_tensors; }
    std::vector<MWNNNode> get_graph_nodes() { return mwnn_nodes; }
    void set_graph_initializers(MWNNTensor m_tensor) { mwnn_initializer_tensors.emplace_back(m_tensor); }
    void set_graph_nodes(MWNNNode m_node) { mwnn_nodes.emplace_back(m_node); }
  private:
    MWNNModel mwnn_model;
    std::string name;
    std::vector<MWNNTensor> mwnn_initializer_tensors;
    std::vector<MWNNNode> mwnn_nodes;
};

} //namespace metawarenn
#endif //METAWARENN_GRAPH_H_
