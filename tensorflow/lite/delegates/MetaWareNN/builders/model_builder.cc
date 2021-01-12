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

void convert_CHWN_to_NHWC(::metawarenn::MWNNGraph *mwnn_graph, std::string initializer_name)
{
  auto weight = mwnn_graph->get_initializer_tensor(initializer_name);
  auto dims = weight.get_dims();
  std::vector<int> new_dims{dims[3], dims[1], dims[2], dims[0]};
  auto tensor = weight.get_tensor();
  std::vector<float> new_wt_buf((dims[0]*dims[1]*dims[2]*dims[3]), 0);
  int channel = dims[3];
  int width = dims[1];
  int height = dims[2];
  for(int i = 0; i < height; i++) {
    for(int j = 0; j < width; j++) {
      for(int k = 0; k < channel; k++) {
        new_wt_buf[(i * width) + (j) +(k * height * width)] = (int16_t)(tensor[(i * width * channel) + (j * channel) + k]);
      }
    }
  }
  mwnn_graph->update_initializer_tensors(weight.get_name(), new_dims, new_wt_buf);
}
/* TODO: High Level Graph to MetaWareNN Graph Representation,
         Apply Passes on MetaWareNN Graph,
         Generate Low Level Graph to run on devices*/
TfLiteStatus ModelBuilder::MetaWareNNCompile(::metawarenn::MWNNGraph *mwnn_graph) {
  std::cout << "\n In MetaWareNNCompile !!! ";
  //Call Passes
  ::metawarenn::optimizer::PassManager manager;
  for (auto node : mwnn_graph->get_graph_nodes())
  {
    //Convert weight layout to common NHWC format before passes
    if(node.get_op_type() == "DepthwiseConv")
    {
      convert_CHWN_to_NHWC(mwnn_graph, node.get_inputs()[1]);
    }
  }
  if(HWC_TO_CHW)
  {
    for (auto g_t : mwnn_graph->get_graph_initializers()) {
      if(g_t.get_dims().size() == 4) {
        std::cout << "\n Name : " << g_t.get_name();
        std::cout << "\t Dims : ";
        for (auto dim : g_t.get_dims())
          std::cout << dim << ",";
        ::metawarenn::optimizer::ConvertLayout cl(mwnn_graph, g_t, 0, HWC_TO_CHW);
        manager.register_pass(cl);
      }
    }
    for (auto g_t : mwnn_graph->get_graph_inputs()) {
      if(g_t.get_dims().size() == 4) {
        std::cout << "\n Name : " << g_t.get_name();
        std::cout << "\t Dims : ";
        for (auto dim : g_t.get_dims())
          std::cout << dim << ",";
        ::metawarenn::optimizer::ConvertLayout cl(mwnn_graph, g_t, 0, HWC_TO_CHW);
        manager.register_pass(cl);
      }
    }
  }
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
