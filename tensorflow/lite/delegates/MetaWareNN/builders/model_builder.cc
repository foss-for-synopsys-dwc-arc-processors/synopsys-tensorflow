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
TfLiteStatus ModelBuilder::MetaWareNNCompile() {
  std::cout << "\n In MetaWareNNCompile !!! ";
  //Call Passes
  ::metawarenn::optimizer::PassManager manager;
  ::metawarenn::optimizer::DummyPass1 d_pass1(7);
  std::cout << "\n MetaWareNNCC : " << d_pass1.get_name();
  ::metawarenn::optimizer::DummyPass2 d_pass2;
  std::cout << "\n MetaWareNNCC : " << d_pass2.get_name();
  ::metawarenn::optimizer::DummyPass3 d_pass3;
  std::cout << "\n MetaWareNNCC : " << d_pass3.get_name();
  ::metawarenn::optimizer::DummyPass1 d_pass4;
  std::cout << "\n MetaWareNNCC : " << d_pass4.get_name();
  manager.register_pass(d_pass1);
  manager.register_pass(d_pass2);
  manager.register_pass(d_pass3);
  manager.register_pass(d_pass4);
  manager.run_passes();
  return kTfLiteOk;
  }
} // namespace metawarenn
} // namespace delegates
} // namespace tflite
