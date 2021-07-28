#include "fuse_batchnorm.h"

namespace metawarenn {

namespace optimizer {

FuseBatchNorm::FuseBatchNorm() {
  set_name("FuseBatchNorm");
}
FuseBatchNorm::FuseBatchNorm(std::shared_ptr<MWNNGraph> mwnn_graph, MWNNNode mwnn_node) {
  set_name("FuseBatchNorm");
  graph = mwnn_graph;
  node = mwnn_node;
}
void FuseBatchNorm::RunPass() {
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

          std::vector<std::string> conv_inputs = g_n.get_inputs();
          std::vector<std::string> bn_inputs = node.get_inputs();

          auto weight = graph->get_initializer_tensor(conv_inputs[1]);
          auto w_tensor = weight.get_tensor();
          auto w_dims = weight.get_dims();

          auto gamma = graph->get_initializer_tensor(bn_inputs[1]);
          auto g_tensor = gamma.get_tensor();
          auto g_dims = gamma.get_dims();

          auto beta = graph->get_initializer_tensor(bn_inputs[2]);
          auto b_tensor = beta.get_tensor();
          auto b_dims = beta.get_dims();

          auto mean = graph->get_initializer_tensor(bn_inputs[3]);
          auto m_tensor = mean.get_tensor();
          auto m_dims = mean.get_dims();

          auto variance = graph->get_initializer_tensor(bn_inputs[4]);
          auto v_tensor = variance.get_tensor();
          auto v_dims = variance.get_dims();

          auto epsilon = node.get_attribute_value_float("epsilon")[0];

          int num_weights = std::accumulate(std::begin(w_dims), std::end(w_dims), 1, std::multiplies<double>());

          int filter = w_dims[0];
          int channel = w_dims[1];
          int height = w_dims[2];
          int width = w_dims[3];
          std::vector<float> merged_weights(num_weights);
          std::vector<float> merged_bias(filter);

          for (int f = 0; f < filter; f++) {
            float multiply_value = g_tensor[f] / sqrt(v_tensor[f] + epsilon);
            merged_bias[f] = (multiply_value * (0 - m_tensor[f])) + b_tensor[f];
            for (int c = 0; c < channel; c++) {
              for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                  int index = (f * channel * height * width) + (c * height * width) + (h * width) + (w);
                   merged_weights[index] = w_tensor[index] * multiply_value;
                }
              }
            }
          }
          std::string bias_name = conv_inputs[1].append("_bias");

          //Update BatchNorm merged weights in Conv
          graph->update_initializer_tensors(weight.get_name(), w_dims, merged_weights);
          //Add BatchNorm merged bias as new initializer in graph
          graph->add_initializer_tensor(bias_name, b_dims, merged_bias);
          //Update BatchNorm merged bias as new input in Conv
          graph->update_node_inputs(g_n.get_name(), bias_name, 2, true);
          //Remove initializer tensor from graph
          graph->remove_initializer_tensor(bn_inputs[1]);
          graph->remove_initializer_tensor(bn_inputs[2]);
          graph->remove_initializer_tensor(bn_inputs[3]);
          graph->remove_initializer_tensor(bn_inputs[4]);
          //Remove initializer names from graph
          graph->remove_initializer_names(bn_inputs[1]);
          graph->remove_initializer_names(bn_inputs[2]);
          graph->remove_initializer_names(bn_inputs[3]);
          graph->remove_initializer_names(bn_inputs[4]);
          //Update Conv Node output name with BatchNorm node output from graph
          graph->update_node_outputs(g_n.get_name(), node.get_outputs()[0], i);
          //Remove BatchNorm Node from graph
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
