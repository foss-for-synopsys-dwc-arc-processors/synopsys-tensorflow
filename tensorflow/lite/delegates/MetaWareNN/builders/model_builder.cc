#include "model_builder.h"

namespace tflite {
namespace delegates {
namespace metawarenn {

ModelBuilder::ModelBuilder(std::vector<int> nodes)
    : subgraph_nodes_(nodes) {}

std::shared_ptr<::metawarenn::MWNNGraph> ModelBuilder::BuildGraph(TfLiteContext* context) {
  std::cout<<"\nBuildGraph!!"<<std::endl;

  /*Create MetaWareNN High Level Graph Representation from TFLite SubGraph Nodes*/
  std::shared_ptr<::metawarenn::MWNNGraph> mwnn_graph_ptr = std::make_shared<::metawarenn::MWNNGraph>(context, subgraph_nodes_);
  return mwnn_graph_ptr;
}

void convert_CHWN_to_NHWC(std::shared_ptr<::metawarenn::MWNNGraph> mwnn_graph, std::string initializer_name)
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
        new_wt_buf[(i * width) + (j) +(k * height * width)] = tensor[(i * width * channel) + (j * channel) + k];
      }
    }
  }
  mwnn_graph->update_initializer_tensors(weight.get_name(), new_dims, new_wt_buf);
}
/* TODO: High Level Graph to MetaWareNN Graph Representation,
         Apply Passes on MetaWareNN Graph,
         Generate Low Level Graph to run on devices*/
TfLiteStatus ModelBuilder::MetaWareNNCompile(std::shared_ptr<::metawarenn::MWNNGraph> mwnn_graph) {
  std::cout << "\n In MetaWareNNCompile !!! ";
  namespace bip = boost::interprocess;
  bip::shared_memory_object::remove("SharedMemoryFile");
  bip::shared_memory_object shm(bip::create_only, "SharedMemoryFile", bip::read_write);
  shm.truncate(60u<<20); // 60MiB
  bip::mapped_region region(shm, bip::read_write);
  bip::bufferstream bs(std::ios::out);
  bs.buffer(reinterpret_cast<char*>(region.get_address()), region.get_size());
  boost::archive::text_oarchive oa(bs);
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
        for(auto node: mwnn_graph->get_graph_nodes()) {
          if(g_t.get_name() == node.get_inputs()[1] && node.get_op_type() == "Conv") {
            std::cout << "\t Dims : ";
            for (auto dim : g_t.get_dims())
              std::cout << dim << ",";
            ::metawarenn::optimizer::ConvertLayout cl(mwnn_graph, g_t, 0, HWC_TO_CHW);
            manager.register_pass(cl);
          }
        }
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
  #if INVOKE_NNAC
    std::cout << "\n ---------------------------Graph----------------------------- \n";
    std::cout << "\n Graph Name : " << mwnn_graph->get_name();
    ::MWNN::MWNNGraphProto mwnn_graph_proto;
    mwnn_graph_proto.set_name(mwnn_graph->get_name());
    mwnn_graph_proto.set_graph_input(mwnn_graph->get_graph_ip_name());
    mwnn_graph_proto.set_graph_output(mwnn_graph->get_graph_op_name());

    std::cout << "\n -----------------------Graph Inputs-------------------------- \n";
    for (auto g_ip : mwnn_graph->get_graph_inputs()) {
      std::cout << "\n Input Name : " << g_ip.get_name();
      std::cout << "\n Data Type : " << g_ip.get_type();
      std::cout << "\n Input Dims : ";
      auto input = mwnn_graph_proto.add_input();
      input->set_name(g_ip.get_name());
      input->set_type(g_ip.get_type());

      for (auto dim : g_ip.get_dims())
        input->add_dims(dim);
    }
    std::cout << "\n -----------------------Graph Outputs-------------------------- \n";
    for (auto g_op : mwnn_graph->get_graph_outputs()) {
      std::cout << "\n Output Name : " << g_op.get_name();
      std::cout << "\n Data Type : " << g_op.get_type();
      std::cout << "\n Output Dims : ";
      auto output = mwnn_graph_proto.add_output();
      output->set_name(g_op.get_name());
      output->set_type(g_op.get_type());
      for (auto dim : g_op.get_dims())
        output->add_dims(dim);

    }
    std::cout << "\n -----------------------Graph Nodes-------------------------- \n";
    for (auto g_n : mwnn_graph->get_graph_nodes()) {
      std::cout << "\n ================================================================ \n";
      std::cout << "\n Node Name : " << g_n.get_name();
      std::cout << "\n Op Type : " << g_n.get_op_type();
      auto node = mwnn_graph_proto.add_node();
      node->set_name(g_n.get_name());
      auto op_type = g_n.get_op_type();
      node->set_op_type(op_type == "DepthwiseConv" ? "Conv" : op_type);
      for (auto n_ip : g_n.get_inputs())
        node->add_input((n_ip));
      for (auto n_op : g_n.get_outputs())
        node->add_output((n_op));
      std::cout << "\n ---------------------------------------------------------------- ";
      for (auto attribute : g_n.get_attributes()) {
        std::cout << "\n Attribute Name : " << attribute.get_name();
        std::cout << "\n Attribute Data Type : " << attribute.get_type();
        std::cout << "\n Attribute Data : ";
        auto attr = node->add_attribute();
        attr->set_name(attribute.get_name());
        attr->set_type(attribute.get_type());
        if(attribute.get_type() == 3 || attribute.get_type() == 8) {
          for(int i = 0; i < attribute.get_string_data().size(); i++){
            auto data = attr->add_data();
            data = &attribute.get_string_data()[i];}
        }
        else {
          for(int i = 0; i < attribute.get_data().size(); i++){
            std::cout << attribute.get_data()[i] << ",";
            attr->add_ints(attribute.get_data()[i]);
          }
        }
      }
    }
    std::cout << "\n -----------------------Graph Tensors-------------------------- \n";
    for (auto g_t : mwnn_graph->get_graph_initializers()) {
      auto initializer = mwnn_graph_proto.add_initializer();
      initializer->set_name(g_t.get_name());
      initializer->set_data_type(g_t.get_type());
      std::cout << "\n Name : " << g_t.get_name();
      std::cout << "\n Type : " << g_t.get_type();
      std::cout << "\n Dim : ";
      for (auto dim : g_t.get_dims()){
        std::cout << dim << ",";
        initializer->add_dims(dim);
      }
      for (auto t_val : g_t.get_tensor())
        initializer->add_float_data(t_val);
    }
    int fp = open("/path/to/store/mobilenetv2-7_graphproto.bin", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    std::cout << fp;
    std::cout << mwnn_graph_proto.SerializeToFileDescriptor(fp);
    close(fp);

    std::cout << "\n\n==============Initiating NNAC python script through shell script=========================\n";
    system("bash /path/to/synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/mwnnconvert/mwnn_convert.sh");
    //exit(1);
  #endif

  //oa << *mwnn_graph;*/
  return kTfLiteOk;
  }
} // namespace metawarenn
} // namespace delegates
} // namespace tflite
