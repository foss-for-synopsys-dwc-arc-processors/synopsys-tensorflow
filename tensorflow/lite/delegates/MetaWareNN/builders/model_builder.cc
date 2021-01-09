#include "model_builder.h"

namespace tflite {
namespace delegates {
namespace metawarenn {

ModelBuilder::ModelBuilder(std::vector<int> nodes)
    : subgraph_nodes_(nodes) {}

::metawarenn::MWNNGraph ModelBuilder::BuildGraph(TfLiteContext* context) {
  std::cout<<"\nBuildGraph!!"<<std::endl;

  /*Create MetaWareNN High Level Graph Representation from TFLite SubGraph Nodes*/
  ::metawarenn::MWNNModel mwnn_model_();
  ::metawarenn::MWNNGraph mwnn_graph_(context, subgraph_nodes_);
  return mwnn_graph_;
}

/* TODO: High Level Graph to MetaWareNN Graph Representation,
         Apply Passes on MetaWareNN Graph,
         Generate Low Level Graph to run on devices*/
TfLiteStatus ModelBuilder::MetaWareNNCompile(::metawarenn::MWNNGraph *mwnn_graph) {
  std::cout << "\n In MetaWareNNCompile !!! ";
  //Call Passes
  ::metawarenn::optimizer::PassManager manager;
  /*
  ::metawarenn::optimizer::DummyPass1 d_pass1(7);
  std::cout << "\n MetaWareNNCC : " << d_pass1.get_name();
  manager.register_pass(d_pass1);*/
  auto node_list = mwnn_graph->get_graph_nodes();
  for (int node_idx = 0; node_idx < mwnn_graph->get_graph_nodes().size() ; node_idx++) {
    auto g_n = node_list[node_idx];
    if(g_n.get_op_type() == "Reshape") {
      ::metawarenn::optimizer::RemoveReshape rr(mwnn_graph, g_n);
      std::cout << "\n MetaWareNNCC : " << rr.get_name();
      manager.register_pass(rr);
    }
    else if(g_n.get_op_type() == "Relu") {
      ::metawarenn::optimizer::FuseRelu fr(mwnn_graph, g_n);
      std::cout << "\n MetaWareNNCC : " << fr.get_name();
      manager.register_pass(fr);
    }
  }
  manager.run_passes();
  return kTfLiteOk;
  }
} // namespace metawarenn
} // namespace delegates
} // namespace tflite
