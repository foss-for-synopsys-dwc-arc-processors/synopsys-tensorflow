#ifndef METAWARENN_GRAPH_H_
#define METAWARENN_GRAPH_H_

#include "metawarenn_model.h"
#include "metawarenn_tensor.h"
#include "metawarenn_node.h"
#include "metawarenn_value_info.h"
#include "op/node.h"
#include <set>
#include <map>

namespace metawarenn {

class MWNNGraph {
  public:
     MWNNGraph() = default;

    void set_name(std::string m_name) { name = m_name; }
    std::string get_name() { return name; }

    void set_graph_ip_name(std::string m_ip_name) { ip_name = m_ip_name; }
    std::string get_graph_ip_name() { return ip_name; }

    void set_graph_op_name(std::string m_op_name) { op_name = m_op_name; }
    std::string get_graph_op_name() { return op_name; }

    void set_graph_initializers(MWNNTensor m_tensor) { mwnn_initializer_tensors.emplace_back(m_tensor); }
    std::vector<MWNNTensor> get_graph_initializers() { return mwnn_initializer_tensors; }

    void set_graph_nodes(MWNNNode m_node) { mwnn_nodes.emplace_back(m_node); }
    std::vector<MWNNNode> get_graph_nodes() { return mwnn_nodes; }

    void set_graph_inputs(MWNNValueInfo m_valueinfo) { mwnn_inputs.emplace_back(m_valueinfo); }
    std::vector<MWNNValueInfo> get_graph_inputs() { return mwnn_inputs; }

    void set_graph_outputs(MWNNValueInfo m_valueinfo) { mwnn_outputs.emplace_back(m_valueinfo); }
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
