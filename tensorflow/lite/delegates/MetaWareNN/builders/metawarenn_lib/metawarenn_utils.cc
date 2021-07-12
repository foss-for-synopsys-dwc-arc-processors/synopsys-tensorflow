#include "metawarenn_utils.h"

namespace metawarenn {

void fill_mwnn_tensor_initalizer(std::string input_name, MWNNGraph mwnn_graph, mli_tensor *mwnn_initalizer, int *k_height, int *k_width, int *ch, int is_HWC)
{
  mwnn_initalizer->el_type = MLI_EL_FX_16;
  auto weight = mwnn_graph.get_initializer_tensor(input_name);
  auto dims = weight.get_dims();
  mwnn_initalizer->rank = dims.size();
  std::copy(dims.begin(), dims.end(), mwnn_initalizer->shape);
  auto tensor = weight.get_tensor();
  auto abs_max = std::abs(*std::max_element(tensor.begin(), tensor.end()));
  auto abs_min = std::abs(*std::min_element(tensor.begin(), tensor.end()));
  auto max = std::max(abs_max, abs_min);
  mwnn_initalizer->el_params.fx.frac_bits = mwnn_initalizer->el_type - (int)ceil(log2(max)) - 1;
  int wt_buf_size = 1;
  uint8_t i;

  if (dims.size() > 1)
  {
    *ch = dims[0];
    *k_height = is_HWC ? dims[1] : dims[2];
    *k_width = is_HWC ? dims[2] : dims[3];
  }

  for (i = 0; i < dims.size(); i++)
  {
    mwnn_initalizer->mem_stride[i] = 0;
    wt_buf_size = wt_buf_size * dims[i];
  }
  int16_t *buffer = (int16_t*)malloc(wt_buf_size * sizeof(int16_t));
  int j = 0;
  for(std::vector<float>::iterator it = tensor.begin(); it != tensor.end(); ++it)
  {
    buffer[j++] = (int16_t)(*it * (1 << (mwnn_initalizer->el_params.fx.frac_bits)) + ((*it >= 0)? 0.5f: -0.5f));
  }
  mwnn_initalizer->data.capacity = sizeof(buffer);
  mwnn_initalizer->data.mem.void_p = (void *)buffer;
}

void fill_mwnn_tensor_input(MWNNTensor input, mli_tensor *mwnn_tensor)
{
  auto dims = input.get_dims();
  mwnn_tensor->rank = dims.size();
  std::copy(dims.begin()+1, dims.end(), mwnn_tensor->shape);
  mwnn_tensor->shape[3] = 1;
  mwnn_tensor->el_type = MLI_EL_FX_16;
  mwnn_tensor->el_params.fx.frac_bits = 8;
  uint8_t i;
  int total_size = 1;
  for (i = 0; i < dims.size(); i++)
  {
    total_size = total_size * dims[i];
    mwnn_tensor->mem_stride[i] = 0;
  }
  auto ip_tensor = input.get_tensor();
  int16_t *input_buffer = (int16_t*)malloc(total_size * sizeof(int16_t));
  for(int i = 0; i < total_size; i++)
  {
    input_buffer[i] = ip_tensor[i];
  }
  mwnn_tensor->data.capacity = total_size * sizeof(int16_t);
  mwnn_tensor->data.mem.void_p = (void *)input_buffer;
}

void create_mwnn_tensor_output(mli_tensor *mwnn_tensor, long int buf_size)
{
  for(int dim = 0; dim < 4; dim++)
    mwnn_tensor->mem_stride[dim] = 0;
  int16_t *out_buffer = (int16_t*)malloc(buf_size * sizeof(int16_t));
  mwnn_tensor->data.capacity = buf_size * sizeof(int16_t);
  mwnn_tensor->data.mem.void_p = (void *)out_buffer;
  mwnn_tensor->el_params.fx.frac_bits = 8;
}

void convert_to_mwnn_format(MWNNGraph mwnn_graph, std::unordered_map<std::string, float*> &graph_inputs, std::unordered_map<std::string, float*> &graph_outputs, int is_HWC)
{
  std::map<std::string, mli_tensor> tensor_map;
  //Fill the MWNNGraph ip tensor
  mwnn_graph.update_input_tensors(graph_inputs);
  std::cout << "\n======================================================================================================================= \n";
  std::cout << "\n --------------------------------- Conversion to MetaWareNN High Level Graph Format -----------------------------------\n";
  auto node_list = mwnn_graph.get_graph_nodes();
  for (int node_idx = 0; node_idx < mwnn_graph.get_graph_nodes().size() ; node_idx++) {
    std::cout << "\n======================================================================================================================= \n";
    auto g_n = node_list[node_idx];
    std::string output_name;
    std::cout << "\nNode name : " << g_n.get_name();
    std::string op_type = g_n.get_op_type();
    if (op_type == "Conv" || op_type == "DepthwiseConv")
    {
      mli_conv2d_cfg conv_cfg;
      int kernel_height, kernel_width, channels;
      auto strides = g_n.get_attribute_value("strides");
      auto pads = g_n.get_attribute_value("pads");
      auto dilations = g_n.get_attribute_value("dilations");
      conv_cfg.stride_height = strides[0];
      conv_cfg.stride_width = strides[1];
      conv_cfg.padding_top = pads[0];
      conv_cfg.padding_left = pads[1];
      conv_cfg.padding_bottom = pads[2];
      conv_cfg.padding_right = pads[3];
      conv_cfg.dilation_height = dilations[0];
      conv_cfg.dilation_width = dilations[1];
      auto activation = g_n.get_attribute_value("activation")[0];
      if(activation == ActivationType::Activation_None)
        conv_cfg.relu.type = MLI_RELU_NONE;
      else if(activation == ActivationType::Activation_Relu)
        conv_cfg.relu.type = MLI_RELU_GEN;
      else if(activation == ActivationType::Activation_Relu6)
        conv_cfg.relu.type = MLI_RELU_6;
      mli_tensor input_tensor;
      mli_tensor conv_wt;
      mli_tensor conv_bias;
      mli_tensor output_tensor;
      std::vector<std::string> inputs = g_n.get_inputs();
      if(inputs.size() == 3)
        fill_mwnn_tensor_initalizer(inputs[2], mwnn_graph, &conv_bias, &kernel_height, &kernel_width, &channels, is_HWC);
      fill_mwnn_tensor_initalizer(inputs[1], mwnn_graph, &conv_wt, &kernel_height, &kernel_width, &channels, is_HWC);
      auto input = g_n.get_inputs()[0];
      // Handles the initial graph input to the first conv node and updates tensor map
      if(input == mwnn_graph.get_graph_ip_name())
      {
        fill_mwnn_tensor_input(mwnn_graph.get_graph_ip_tensor()[0], &input_tensor);
        tensor_map.insert(std::pair<std::string, mli_tensor>(input, input_tensor));
      }
      input_tensor = (tensor_map.find(input))->second;
      // Output buffer size calculation
      int input_height = is_HWC ? input_tensor.shape[0]: input_tensor.shape[1];
      int input_width = is_HWC ? input_tensor.shape[1] : input_tensor.shape[2];
      int effective_kernel_width = (kernel_width - 1) * conv_cfg.dilation_width + 1;
      int effective_kernel_height = (kernel_height - 1) * conv_cfg.dilation_height + 1;
      const int out_width  = CEIL_DIV(input_width + conv_cfg.padding_left + conv_cfg.padding_right - effective_kernel_width + 1,
                                      conv_cfg.stride_width);
      const int out_height = CEIL_DIV(input_height + conv_cfg.padding_top + conv_cfg.padding_bottom - effective_kernel_height + 1,
                                      conv_cfg.stride_height);
      create_mwnn_tensor_output(&output_tensor, out_width * out_height * channels);
      // General convolution invocation
      if(op_type == "Conv")
      {
        if(is_HWC)
          mli::krn::ref::conv2d_prepare_and_run<int16_t, int16_t, int16_t, mli_fx16_accu_t, mli::krn::fx_quant_specific_params, LAYOUT_HWC,  mli::CONV_GENERAL>(
            &(tensor_map.find(input))->second,
            &conv_wt,
            &conv_bias,
            &conv_cfg, &output_tensor);
        else
          mli::krn::ref::conv2d_prepare_and_run<int16_t, int16_t, int16_t, mli_fx16_accu_t, mli::krn::fx_quant_specific_params, LAYOUT_CHW,  mli::CONV_GENERAL>(
            &(tensor_map.find(input))->second,
            &conv_wt,
            &conv_bias,
            &conv_cfg, &output_tensor);
      }
      // Depthwise convolution invocation
      else if(op_type == "DepthwiseConv")
      {
        if(is_HWC)
          mli::krn::ref::conv2d_prepare_and_run<int16_t, int16_t, int16_t, mli_fx16_accu_t, mli::krn::fx_quant_specific_params, LAYOUT_HWC,  mli::CONV_DEPTHWISE>(
            &(tensor_map.find(input))->second,
            &conv_wt,
            &conv_bias,
            &conv_cfg, &output_tensor);
        else
          mli::krn::ref::conv2d_prepare_and_run<int16_t, int16_t, int16_t, mli_fx16_accu_t, mli::krn::fx_quant_specific_params, LAYOUT_CHW,  mli::CONV_DEPTHWISE>(
            &(tensor_map.find(input))->second,
            &conv_wt,
            &conv_bias,
            &conv_cfg, &output_tensor);
      }

      output_tensor.shape[3] = 1;
      tensor_map.insert(std::pair<std::string, mli_tensor>(g_n.get_outputs()[0], output_tensor)); // Store the output tensor to tensor map
    }
    else if (op_type =="Add")
    {
      mli_tensor output_tensor;
      auto input = g_n.get_inputs();
      auto shape = (tensor_map.find(input[0]))->second.shape;
      int buf_size = 1;
      int rank = (tensor_map.find(input[0]))->second.rank;
      for (int i = 0; i < rank; i++)
      {
        buf_size = buf_size * shape[i];
      }

      // To change data layout from CHW to NCHW
      int temp = (tensor_map.find(input[0]))->second.shape[rank - 1];
      for(int i = (tensor_map.find(input[0]))->second.rank - 1; i > 0; i--)
      {
          (tensor_map.find(input[0]))->second.shape[i] = shape[i-1];
      }
      (tensor_map.find(input[0]))->second.shape[0] = temp;

      create_mwnn_tensor_output(&output_tensor, buf_size);
      mli::krn::ref::eltwise_prepare_and_run<int16_t, mli::ELTWISE_ADD>(&(tensor_map.find(input[0]))->second, &(tensor_map.find(input[1]))->second, &output_tensor);
      tensor_map.insert(std::pair<std::string, mli_tensor>(g_n.get_outputs()[0], output_tensor));
    }
    else if (op_type =="GlobalAveragePool")
    {
      mli_tensor output_tensor;
      auto input = g_n.get_inputs();
      auto shape = (tensor_map.find(input[0]))->second.shape;
      int buf_size = 1;
      int rank = (tensor_map.find(input[0]))->second.rank;
      for (int i = 0; i < rank; i++)
      {
        buf_size = buf_size * shape[i];
      }
      mli_tensor input_tensor = (tensor_map.find(input[0]))->second;
      int16_t *input_buf, *new_input_buf;
      int channel, width, height;
      if(!(is_HWC)) //CHW
      {
        // Data layout conversion from CHW to HWC
        input_buf = (int16_t *)(tensor_map.find(input[0]))->second.data.mem.void_p;//chw
        new_input_buf = (int16_t*)malloc(buf_size * sizeof(int16_t));//hwc
        channel = (tensor_map.find(input[0]))->second.shape[FMAP_C_DIM_CHW];
        width = (tensor_map.find(input[0]))->second.shape[FMAP_W_DIM_CHW];
        height = (tensor_map.find(input[0]))->second.shape[FMAP_H_DIM_CHW];

        for (int i = 0; i < channel; i++) {
          for(int j = 0; j < height; j++) {
            for(int k = 0; k < width; k++) {
              new_input_buf[i + (j * width * channel) + (k * channel)] = input_buf[(i * height * width) + (j * width) + k];
            }
          }
        }
        (tensor_map.find(input[0]))->second.data.mem.void_p = (void*)new_input_buf;
        (tensor_map.find(input[0]))->second.shape[FMAP_H_DIM_HWC] = height;
        (tensor_map.find(input[0]))->second.shape[FMAP_W_DIM_HWC] = width;
        (tensor_map.find(input[0]))->second.shape[FMAP_C_DIM_HWC] = channel;
      }
      create_mwnn_tensor_output(&output_tensor, buf_size);

      mli_pool_cfg pool_cfg;
      pool_cfg.kernel_width = is_HWC ? input_tensor.shape[1] : input_tensor.shape[2];
      pool_cfg.kernel_height = is_HWC ? input_tensor.shape[0] : input_tensor.shape[1];
      pool_cfg.stride_width = 1;
      pool_cfg.stride_height = 1;
      pool_cfg.padding_top = 0;
      pool_cfg.padding_bottom = 0;
      pool_cfg.padding_left = 0;
      pool_cfg.padding_right = 0;

      mli::krn::mli_krn_avepool_hwc<int16_t, mli_fx16_accu_t, 0>(&(tensor_map.find(input[0]))->second, &pool_cfg, &output_tensor);

      if(!is_HWC) //CHW
      {
      // Data layout conversion from HWC to CHW
        channel = output_tensor.shape[FMAP_C_DIM_HWC];
        width = output_tensor.shape[FMAP_W_DIM_HWC];
        height = output_tensor.shape[FMAP_H_DIM_HWC];
        input_buf = (int16_t *)output_tensor.data.mem.void_p;//hwc
        new_input_buf = (int16_t*)malloc(buf_size * sizeof(int16_t));//chw

        for(int i = 0; i < height; i++) {
          for(int j = 0; j < width; j++) {
            for(int k = 0; k < channel; k++) {
              new_input_buf[(i * width) + (j) +(k * height * width)] = (int16_t)(input_buf[(i * width * channel) + (j * channel) + k]);
            }
          }
        }
        output_tensor.data.mem.void_p = (void*)new_input_buf;
        output_tensor.shape[FMAP_H_DIM_CHW] = height;
        output_tensor.shape[FMAP_W_DIM_CHW] = width;
        output_tensor.shape[FMAP_C_DIM_CHW] = channel;
      }
      tensor_map.insert(std::pair<std::string, mli_tensor>(g_n.get_outputs()[0], output_tensor));
    }
    // MLI kernel invocation yet to be handled.
    else if (op_type =="Reshape")
    {
      auto input = g_n.get_inputs();
      tensor_map.insert(std::pair<std::string, mli_tensor>(g_n.get_outputs()[0], (tensor_map.find(input[0]))->second));
    }
    else if (op_type =="Softmax")
    {
      auto input = g_n.get_inputs();
      tensor_map.insert(std::pair<std::string, mli_tensor>(g_n.get_outputs()[0], (tensor_map.find(input[0]))->second));
    }
    else if(op_type == "MaxPool")
    {}
    else if(op_type == "Gemm")
    {}
    else if(op_type == "Flatten")
    {}
    else if(op_type == "BatchNormalization")
    {}
    else if(op_type == "Concat")
    {}
    else if(op_type == "LRN")
    {}
    else if(op_type == "Mul")
    {}
    else if(op_type == "Transpose")
    {}
    else if(op_type == "AvgPool")
    {}
    else if(op_type == "Clip")
    {}
    else if(op_type == "Shape")
    {}
    else if(op_type == "Squeeze")
    {}
    else if(op_type == "Unsqueeze")
    {}
    else if(op_type == "Mean")
    {}
    else if(op_type == "Split")
    {}
    else if(op_type == "Pad")
    {}
    else if(op_type == "StridedSlice")
    {}
    else if(op_type == "Mean")
    {}
    else if(op_type == "FullyConnected")
    {}
  }
  //To fill the graph output layer value in the passed argument
  for (auto g_op : graph_outputs) {
    for (auto t_map : tensor_map) {
      if(t_map.first == g_op.first) {
        float *op = (float *)((t_map.second).data.mem.void_p);
        graph_outputs[g_op.first] = op;
        break;
      }
    }
  }
  //Fill the MWNNGraph op tensor
  mwnn_graph.update_output_tensors(graph_outputs);
}

} //namespace metawarenn
