#ifndef METAWARENN_GRAPH_H_
#define METAWARENN_GRAPH_H_

#include "metawarenn_common.h"
#include "metawarenn_tensor.h"
#include "metawarenn_node.h"
#include "metawarenn_element.h"
#include "metawarenn_value_info.h"
#include "op/node.h"

namespace metawarenn {

class MWNNGraph {
  public:
    MWNNGraph() = default;
    #if ONNX
    MWNNGraph(GraphProto& onnx_graph_proto, std::string graph_name);
    #endif
    #if TFLITE
    MWNNGraph(TfLiteContext* context, std::vector<int> subgraph_nodes_, std::string subgraph_name);
    #endif
    #if GLOW
    MWNNGraph(Function *F, std::string subgraph_name);
    #endif
    #if TVM
    MWNNGraph(std::vector<JSONGraphNode> graph_nodes_, std::string graph_name);
    void set_graph_initializers(std::string const_name, const DLTensor* data);
    void set_graph_inputs(std::string name, const JSONGraphNode& node);
    void set_graph_outputs(std::string name, std::vector<int> dims, int type);
    #endif
    std::string get_name() { return name; }
    std::vector<std::string> get_graph_ip_names() { return mwnn_graph_ip_names; }
    std::vector<std::string> get_graph_op_names() { return mwnn_graph_op_names; }
    void set_graph_op_name(std::string op_name) { mwnn_graph_op_names.emplace_back(op_name); }
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
    void remove_graph_op_names(std::string name){
      auto it = std::find_if(
      std::begin(mwnn_graph_op_names), std::end(mwnn_graph_op_names), [&](std::string op_name) {
          return op_name == name;
      });
      mwnn_graph_op_names.erase(it);
    }
    void remove_initializer_names(std::string name){
      auto it = mwnn_initializer_names.find(name);
      if(it != mwnn_initializer_names.end())
        mwnn_initializer_names.erase(it);
    }
    void remove_initializer_tensor(std::string name){
      auto it = std::find_if(
      std::begin(mwnn_initializer_tensors), std::end(mwnn_initializer_tensors), [&](MWNNTensor& tensor) {
          return tensor.get_name() == name;
      });
      mwnn_initializer_tensors.erase(it);
    }
    void remove_nodes(std::string name){
      auto it = std::find_if(
      std::begin(mwnn_nodes), std::end(mwnn_nodes), [&](MWNNNode& node) {
          return node.get_name() == name;
      });
      mwnn_nodes.erase(it);
    }
     void remove_graph_nodes(std::string name){
      mwnn_graph_nodes.erase(name);
    }
    void update_node_inputs(std::string node_name, std::string ip_name, int index, bool new_ip=false) {
      auto it = std::find_if(
      std::begin(mwnn_nodes), std::end(mwnn_nodes), [&](MWNNNode& node) {
          return node.get_name() == node_name;
      });
      if(new_ip)
        return it->add_inputs(ip_name);
      else
        return it->set_inputs(ip_name, index);
    }
    void update_node_outputs(std::string node_name, std::string op_name, int index) {
      auto it = std::find_if(
      std::begin(mwnn_nodes), std::end(mwnn_nodes), [&](MWNNNode& node) {
          return node.get_name() == node_name;
      });
      return it->set_outputs(op_name, index);
    }
    void update_node_attribute(std::string node_name, std::string attr_name, int value) {
      auto it = std::find_if(
      std::begin(mwnn_nodes), std::end(mwnn_nodes), [&](MWNNNode& node) {
          return node.get_name() == node_name;
      });
      if (it == std::end(mwnn_nodes)) {
          std::cout << "\n ERROR : End of Nodes!!! - Couldn't find node" << name << " to update attribute!!!";
      }
      return it->update_attribute_value(attr_name, value);
    }
    void update_initializer_tensors(std::string tensor_name, std::vector<int> n_dims, std::vector<float> n_tensor) {
      auto it = std::find_if(
      std::begin(mwnn_initializer_tensors), std::end(mwnn_initializer_tensors), [&](MWNNTensor& tensor) {
          return tensor.get_name() == tensor_name;
      });
      return it->update_tensor(n_dims, n_tensor);
    }
    void add_initializer_tensor(std::string tensor_name, std::vector<int> n_dims, std::vector<float> n_tensor) {
      #if TVM
      MWNNTensor mwnn_tensor(tensor_name, n_dims, 2, n_tensor);
      mwnn_initializer_tensors.emplace_back(mwnn_tensor);
      mwnn_initializer_names.insert(mwnn_tensor.get_name());
      auto const_node = mwnn_tensor.get_constant_node();
      mwnn_graph_nodes[mwnn_tensor.get_name()] = std::move(const_node);
      #endif
    }
    void update_initializer_index(std::string tensor_name, uint32_t value) {
      auto it = std::find_if(
      std::begin(mwnn_initializer_tensors), std::end(mwnn_initializer_tensors), [&](MWNNTensor& tensor) {
          return tensor.get_name() == tensor_name;
      });
      return it->set_index(value);
    }
    void update_initializer_offset(std::string tensor_name, uint32_t offset) {
      auto it = std::find_if(
      std::begin(mwnn_initializer_tensors), std::end(mwnn_initializer_tensors), [&](MWNNTensor& tensor) {
          return tensor.get_name() == tensor_name;
      });
      return it->set_offset(offset);
    }
    void update_input_tensors(std::unordered_map<std::string, float*> graph_inputs) {
      for (auto it = mwnn_graph_ip_tensors.begin(); it != mwnn_graph_ip_tensors.end(); ++it) {
        auto name = it->get_name();
        if(graph_inputs.count(name)) {
          float* arr_tensor = graph_inputs[name];
          std::vector<int> dims = it->get_dims();
          int num_elements = std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<double>());
          std::vector<float> vec_tensor(arr_tensor, arr_tensor + num_elements);
          return it->update_tensor(dims, vec_tensor);
        }
      }
    }
    void update_output_tensors(std::unordered_map<std::string, float*> graph_outputs) {
      for (auto it = mwnn_graph_op_tensors.begin(); it != mwnn_graph_op_tensors.end(); ++it) {
        auto name = it->get_name();
        if(graph_outputs.count(name)) {
          float* arr_tensor = graph_outputs[name];
          std::vector<int> dims = it->get_dims();
          int num_elements = std::accumulate(std::begin(dims), std::end(dims), 1, std::multiplies<double>());
          std::vector<float> vec_tensor(arr_tensor, arr_tensor + num_elements);
          return it->update_tensor(dims, vec_tensor);
        }
      }
    }
    void update_inputs(std::string ip_tensor_name, std::vector<int> n_dims) {
      auto it = std::find_if(
      std::begin(mwnn_graph_ip_tensors), std::end(mwnn_graph_ip_tensors), [&](MWNNTensor& tensor) {
          return tensor.get_name() == ip_tensor_name;
      });
      return it->update_dims(n_dims);
    }
    std::vector<MWNNNode> get_graph_nodes() { return mwnn_nodes; }
    std::vector <MWNNTensor> get_graph_ip_tensor() { return mwnn_graph_ip_tensors; }
    std::vector <MWNNTensor> get_graph_op_tensor() { return mwnn_graph_op_tensors; }
    std::set<std::string> mwnn_initializer_names;
    std::map<std::string, std::shared_ptr<op::Node>> mwnn_graph_nodes;
  private:
    std::string name;
    std::vector<std::string> mwnn_graph_ip_names;
    std::vector<std::string> mwnn_graph_op_names;
    std::vector<MWNNNode> mwnn_nodes;
    std::vector<MWNNTensor> mwnn_initializer_tensors;
    std::vector<MWNNTensor> mwnn_graph_ip_tensors;
    std::vector<MWNNTensor> mwnn_graph_op_tensors;
};

} //namespace metawarenn
#endif //METAWARENN_GRAPH_H_
