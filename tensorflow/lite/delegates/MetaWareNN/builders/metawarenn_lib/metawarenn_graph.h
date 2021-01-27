#ifndef METAWARENN_GRAPH_H_
#define METAWARENN_GRAPH_H_

#include "metawarenn_model.h"
#include "metawarenn_tensor.h"
#include "metawarenn_node.h"
#include "metawarenn_element.h"
#include "metawarenn_value_info.h"
#include "op/node.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace metawarenn {

class MWNNGraph {
  public:
    MWNNGraph() = default;
    MWNNGraph(GraphProto& onnx_graph_proto, MWNNModel& model);
    MWNNGraph(TfLiteContext* context, std::vector<int> subgraph_nodes_);
    std::string get_name() { return name; }
    std::string get_graph_ip_name() { return ip_name; }
    std::string get_graph_op_name() { return op_name; }
    void set_graph_op_name(std::string name) { op_name = name; }
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
    void remove_initializer_names(std::string name){
      std::cout << "\n mwnn_initializer_names size before: " << mwnn_initializer_names.size();
      auto it = mwnn_initializer_names.find(name);
      mwnn_initializer_names.erase(it);
      std::cout << "\n mwnn_initializer_names size after: " << mwnn_initializer_names.size();
    }
    void remove_initializer_tensor(std::string name){
      std::cout << "\n mwnn_initializer_tensors size before: " << mwnn_initializer_tensors.size();
      auto it = std::find_if(
      std::begin(mwnn_initializer_tensors), std::end(mwnn_initializer_tensors), [&](MWNNTensor& tensor) {
          return tensor.get_name() == name;
      });
      mwnn_initializer_tensors.erase(it);
      std::cout << "\n mwnn_initializer_tensors size after: " << mwnn_initializer_tensors.size();
    }
    void remove_nodes(std::string name){
      std::cout << "\n mwnn_nodes size before: " << mwnn_nodes.size();
      auto it = std::find_if(
      std::begin(mwnn_nodes), std::end(mwnn_nodes), [&](MWNNNode& node) {
          return node.get_name() == name;
      });
      mwnn_nodes.erase(it);
      std::cout << "\n mwnn_nodes size after: " << mwnn_nodes.size();
    }
     void remove_inputs(std::string name){
      std::cout << "\n mwnn_inputs size before: " << mwnn_inputs.size();
      auto it = std::find_if(
      std::begin(mwnn_inputs), std::end(mwnn_inputs), [&](MWNNValueInfo& valueinfo) {
          return valueinfo.get_name() == name;
      });
      mwnn_inputs.erase(it);
      std::cout << "\n mwnn_inputs size after: " << mwnn_inputs.size();
    }
     void remove_outputs(std::string name){
      std::cout << "\n mwnn_outputs size before: " << mwnn_outputs.size();
      auto it = std::find_if(
      std::begin(mwnn_outputs), std::end(mwnn_outputs), [&](MWNNValueInfo& valueinfo) {
          return valueinfo.get_name() == name;
      });
      mwnn_outputs.erase(it);
      std::cout << "\n mwnn_outputs size after: " << mwnn_outputs.size();
    }
     void remove_graph_nodes(std::string name){
      std::cout << "\n mwnn_graph_nodes size before: " << mwnn_graph_nodes.size();
      mwnn_graph_nodes.erase(name);
      std::cout << "\n mwnn_graph_nodes size after: " << mwnn_graph_nodes.size();
    }
    void update_node_inputs(std::string node_name, std::string ip_name, int index) {
      auto it = std::find_if(
      std::begin(mwnn_nodes), std::end(mwnn_nodes), [&](MWNNNode& node) {
          return node.get_name() == node_name;
      });
      return it->set_inputs(ip_name, index);
    }
    void update_node_outputs(std::string node_name, std::string op_name, int index) {
      auto it = std::find_if(
      std::begin(mwnn_nodes), std::end(mwnn_nodes), [&](MWNNNode& node) {
          return node.get_name() == node_name;
      });
      return it->set_outputs(op_name, index);
    }
    void update_initializer_tensors(std::string tensor_name, std::vector<int> n_dims, std::vector<float> n_tensor) {
      auto it = std::find_if(
      std::begin(mwnn_initializer_tensors), std::end(mwnn_initializer_tensors), [&](MWNNTensor& tensor) {
          return tensor.get_name() == tensor_name;
      });
      return it->update_tensor(n_dims, n_tensor);
    }
    void update_inputs(std::string value_info_name, std::vector<int> n_dims) {
      auto it = std::find_if(
      std::begin(mwnn_inputs), std::end(mwnn_inputs), [&](MWNNValueInfo& valueinfo) {
          return valueinfo.get_name() == value_info_name;
      });
      return it->update_dims(n_dims);
    }
    std::vector<MWNNNode> get_graph_nodes() { return mwnn_nodes; }
    std::vector<MWNNValueInfo> get_graph_inputs() { return mwnn_inputs; }
    std::vector<MWNNValueInfo> get_graph_outputs() { return mwnn_outputs; }

    std::set<std::string> mwnn_initializer_names;
    std::map<std::string, std::shared_ptr<op::Node>> mwnn_graph_nodes;
  private:
    GraphProto graph_proto;
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
