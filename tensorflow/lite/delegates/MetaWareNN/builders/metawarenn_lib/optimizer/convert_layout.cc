#include "convert_layout.h"

namespace metawarenn {

namespace optimizer {

ConvertLayout::ConvertLayout() {
  set_name("ConvertLayout");
}
ConvertLayout::ConvertLayout(std::shared_ptr<MWNNGraph> mwnn_graph, MWNNTensor mwnn_tensor, bool to_hwc, bool to_chw) {
  set_name("ConvertLayout");
  graph = mwnn_graph;
  tensor = mwnn_tensor;
  CHW_to_HWC = to_hwc;
  HWC_to_CHW = to_chw;
  is_tensor = true;
}
ConvertLayout::ConvertLayout(std::shared_ptr<MWNNGraph> mwnn_graph, MWNNValueInfo mwnn_value_info, bool to_hwc, bool to_chw) {
  set_name("ConvertLayout");
  graph = mwnn_graph;
  value_info = mwnn_value_info;
  CHW_to_HWC = to_hwc;
  HWC_to_CHW = to_chw;
  is_value_info = true;
}
void ConvertLayout::RunPass() {
  if(is_tensor) {
    if(CHW_to_HWC) {
      std::vector<int> dims = tensor.get_dims();
      std::vector<int> new_dims{dims[0], dims[2], dims[3], dims[1]};
      std::vector<float> data = tensor.get_tensor();
      std::vector<float> new_data((dims[0]*dims[1]*dims[2]*dims[3]), 0);
      int channel = dims[1];
      int height = dims[2];
      int width = dims[3];
      int num_output = dims[0];
      // Data layout conversion from CHW to HWC

      for(int n = 0; n < num_output; n++) {
        for(int i = 0; i < height; i++) {
          for(int j = 0; j < width; j++) {
            for(int k = 0; k < channel; k++) {
              new_data[(n * height * width * channel) + (i * width * channel) + (j * channel) + k] = data[(n * height * width * channel) + (k * height * width) + (i * width) + (j)];
            }
          }
        }
      }
      graph->update_initializer_tensors(tensor.get_name(), new_dims, new_data);

      auto& node = graph->mwnn_graph_nodes[tensor.get_name()];
      if (auto constant_node = std::dynamic_pointer_cast<::metawarenn::op::Constant>(node)) {
        constant_node->shape = new_dims;
        constant_node->data = new_data;
      }

    }
    else if(HWC_to_CHW) {
      std::vector<int> dims = tensor.get_dims();
      std::vector<int> new_dims{dims[0], dims[3], dims[1], dims[2]};
      std::vector<float> data = tensor.get_tensor();
      std::vector<float> new_data((dims[0]*dims[1]*dims[2]*dims[3]), 0);
      int channel = dims[3];
      int height = dims[1];
      int width = dims[2];
      int num_output = dims[0];
      // Data layout conversion from HWC to CHW
      for(int n = 0; n < num_output; n++) {
        for(int i = 0; i < height; i++) {
          for(int j = 0; j < width; j++) {
            for(int k = 0; k < channel; k++) {
              new_data[(n * height * width * channel) + (k * height * width) + (i * width) + j] = data[(n * height * width * channel) + (i * width * channel) + (j * channel) + k];
            }
          }
        }
      }
      graph->update_initializer_tensors(tensor.get_name(), new_dims, new_data);

      auto& node = graph->mwnn_graph_nodes[tensor.get_name()];
      if (auto constant_node = std::dynamic_pointer_cast<::metawarenn::op::Constant>(node)) {
        constant_node->shape = new_dims;
        constant_node->data = new_data;
      }

    }
  }
  else if(is_value_info) {
    if(CHW_to_HWC) {
      std::vector<int> dims = value_info.get_dims();
      std::vector<int> new_dims{dims[0], dims[2], dims[3], dims[1]};
      graph->update_inputs(value_info.get_name(), new_dims);

      auto& node = graph->mwnn_graph_nodes[value_info.get_name()];
      if (auto input_data_node = std::dynamic_pointer_cast<::metawarenn::op::InputData>(node))
        input_data_node->shape = new_dims;
    }
    else if(HWC_to_CHW) {
      std::vector<int> dims = value_info.get_dims();
      std::vector<int> new_dims{dims[0], dims[3], dims[1], dims[2]};
      graph->update_inputs(value_info.get_name(), new_dims);

      auto& node = graph->mwnn_graph_nodes[value_info.get_name()];
      if (auto input_data_node = std::dynamic_pointer_cast<::metawarenn::op::InputData>(node))
        input_data_node->shape = new_dims;
    }
  }
  }
} //namespace optimizer

} //namespace metawarenn
