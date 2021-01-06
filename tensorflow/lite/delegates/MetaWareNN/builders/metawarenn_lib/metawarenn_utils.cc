
#include "metawarenn_utils.h"
using namespace std;

namespace metawarenn {

void fill_mwnn_tensor_initalizer(std::string input_name, MWNNGraph mwnn_graph, std::string op_type, mli_tensor *mwnn_initalizer, int *k_height, int *k_width, int *ch)
{
  std::cout << "\n\nInitializer name: " << input_name;
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

  std::cout << "\nDimension ssize: ";
  for (i = 0; i < dims.size(); i++)
  {
    mwnn_initalizer->mem_stride[i] = 0;
    std::cout << dims[i] << ", ";
    wt_buf_size = wt_buf_size * dims[i];
  }
  int16_t *buffer = (int16_t*)malloc(wt_buf_size * sizeof(int16_t));
  int j = 0;
  for(std::vector<float>::iterator it = tensor.begin(); it != tensor.end(); ++it)
  {
    buffer[j++] = (int16_t)(*it * (1 << (mwnn_initalizer->el_params.fx.frac_bits)) + ((*it >= 0)? 0.5f: -0.5f));
  }
  if (dims.size() > 1) //To handle weights
  {
    if(op_type == "DepthwiseConv") // To handle DepthwiseConv weights
    {
      // Data layout conversion from CHWN to NHWC
      int16_t *new_wt_buf = (int16_t*)malloc(wt_buf_size * sizeof(int16_t));
      int channel = dims[3];
      int width = dims[1];
      int height = dims[2];
      new_wt_buf = (int16_t*)malloc(wt_buf_size * sizeof(int16_t));

      for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
          for(int k = 0; k < channel; k++) {
            new_wt_buf[(i * width) + (j) +(k * height * width)] = (int16_t)(buffer[(i * width * channel) + (j * channel) + k]);
          }
        }
      }
      // shape conversion for DepthwiseConv weightds
      int temp = mwnn_initalizer->shape[0];
      mwnn_initalizer->shape[0] = dims[3];
      mwnn_initalizer->shape[3] = dims[0];

      mwnn_initalizer->data.capacity = sizeof(new_wt_buf);
      mwnn_initalizer->data.mem.void_p = (void *)new_wt_buf;
    }
    else // To handle conv weights
    {
      mwnn_initalizer->data.capacity = sizeof(buffer);
      mwnn_initalizer->data.mem.void_p = (void *)buffer;
    }
    *ch = mwnn_initalizer->shape[KRNL_C_DIM_HWC];
    *k_height = (int)mwnn_initalizer->shape[KRNL_H_DIM_HWC];
    *k_width = (int)mwnn_initalizer->shape[KRNL_W_DIM_HWC];
  }
  else // To handle bias
  {
  mwnn_initalizer->data.capacity = sizeof(buffer);
  mwnn_initalizer->data.mem.void_p = (void *)buffer;
  }


  std::cout << "\nMax of tensor: " << max;
  std::cout << "\nInt bits : " << (int)ceil(log2(max));
  std::cout << "\nFractional bits : " << (int)mwnn_initalizer->el_params.fx.frac_bits;
  std::cout << "\nInitializer element type : " << mwnn_initalizer->el_type;
  std::cout << "\nInitializer rank : " << mwnn_initalizer->rank;
}

void fill_mwnn_tensor_input(MWNNValueInfo input, mli_tensor *mwnn_tensor)
{
  auto dims = input.get_dims();
  mwnn_tensor->rank = dims.size();
  std::copy(dims.begin()+1, dims.end(), mwnn_tensor->shape);
  mwnn_tensor->shape[3] = 1;
  mwnn_tensor->el_type = MLI_EL_FX_16;
  mwnn_tensor->el_params.fx.frac_bits = 8;
  uint8_t i;
  for (i = 0; i < dims.size(); i++)
  {
    mwnn_tensor->mem_stride[i] = 0;
  }
  int16_t *input_buffer = (int16_t*)malloc(MAX_INPUT_BUF_SIZE * sizeof(int16_t));
  for(int i = 0; i < MAX_INPUT_BUF_SIZE; i++)
  {
    input_buffer[i] = (int16_t)5;
  }
  mwnn_tensor->data.capacity = MAX_INPUT_BUF_SIZE * sizeof(int16_t);
  mwnn_tensor->data.mem.void_p = (void *)input_buffer;
  std::cout << "\nInput's data capacity: " << mwnn_tensor->data.capacity;
}

void create_mwnn_tensor_output(mli_tensor *mwnn_tensor, long int buf_size)
{
  for(int dim = 0; dim < 4; dim++)
    mwnn_tensor->mem_stride[dim] = 0;
  int16_t *out_buffer = (int16_t*)malloc(buf_size * sizeof(int16_t));
  mwnn_tensor->data.capacity = buf_size * sizeof(int16_t);
  mwnn_tensor->data.mem.void_p = (void *)out_buffer;
  mwnn_tensor->el_params.fx.frac_bits = 8;
  std::cout << "\nOutput's data capacity: " << mwnn_tensor->data.capacity;
}

void convert_to_mwnn_format(MWNNGraph mwnn_graph)
{
  std::map<std::string, mli_tensor> tensor_map;
  std::cout << "\n======================================================================================================================= \n";
  std::cout << "\n --------------------------------- Conversion to MetaWareNN High Level Graph Format -----------------------------------\n";
  auto node_list = mwnn_graph.get_graph_nodes();
  for (int node_idx =0; node_idx < mwnn_graph.get_graph_nodes().size() ; node_idx++) {
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
      std::cout << "\n\nConfig params:";
      std::cout << "\nstride_height : " << (int)conv_cfg.stride_height;
      std::cout << "\nstride_width : " << (int)conv_cfg.stride_width;
      std::cout << "\npadding_top : " << (int)conv_cfg.padding_top;
      std::cout << "\npadding_left : " << (int)conv_cfg.padding_left;
      std::cout << "\npadding_bottom : " << (int)conv_cfg.padding_bottom;
      std::cout << "\npadding_right : " << (int)conv_cfg.padding_right;
      std::cout << "\ndilation_height : " << (int)conv_cfg.dilation_height;
      std::cout << "\ndilation_width : " << (int)conv_cfg.dilation_width;
      // Get the next relu node to fuse with conv+BN and update the output_name to fetch as i/p for next conv node
      auto next_node = node_list[node_idx+1];
      if (next_node.get_op_type() == "Relu")
      {
        output_name = next_node.get_name();
        conv_cfg.relu.type = MLI_RELU_GEN;
      }
      else
      {
        output_name = g_n.get_outputs()[0];
        conv_cfg.relu.type = MLI_RELU_NONE;
      }
      mli_tensor input_tensor;
      mli_tensor conv_wt;
      mli_tensor conv_bias;
      mli_tensor output_tensor;
      std::vector<std::string> inputs = g_n.get_inputs();
      if(inputs.size() == 3)
        fill_mwnn_tensor_initalizer(inputs[2], mwnn_graph, op_type, &conv_bias, &kernel_height, &kernel_width, &channels);
      fill_mwnn_tensor_initalizer(inputs[1], mwnn_graph, op_type, &conv_wt, &kernel_height, &kernel_width, &channels);
      auto input = g_n.get_inputs()[0];
      // Handles the initial graph input to the first conv node and updates tensor map
      if(input == mwnn_graph.get_graph_ip_name())
      {
        fill_mwnn_tensor_input(mwnn_graph.get_graph_inputs()[0], &input_tensor);
        tensor_map.insert(std::pair<std::string, mli_tensor>(input, input_tensor));
      }
      // Output buffer size calculation
      int input_height = (tensor_map.find(input))->second.shape[0];
      int input_width = (tensor_map.find(input))->second.shape[1];
      int effective_kernel_width = (kernel_width - 1) * conv_cfg.dilation_width + 1;
      int effective_kernel_height = (kernel_height - 1) * conv_cfg.dilation_height + 1;
      const int out_width  = CEIL_DIV(input_width + conv_cfg.padding_left + conv_cfg.padding_right - effective_kernel_width + 1,
                                      conv_cfg.stride_width);
      const int out_height = CEIL_DIV(input_height + conv_cfg.padding_top + conv_cfg.padding_bottom - effective_kernel_height + 1,
                                      conv_cfg.stride_height);
      create_mwnn_tensor_output(&output_tensor, out_width * out_height * channels);
      std::cout << "\nInput node: " << input;
      // General convolution invocation
      if(op_type == "Conv")
      {
        mli::krn::ref::conv2d_prepare_and_run<int16_t, int16_t, int16_t, mli_fx16_accu_t, mli::krn::fx_quant_specific_params, LAYOUT_HWC,  mli::CONV_GENERAL>(
          &(tensor_map.find(input))->second,
          &conv_wt,
          &conv_bias,
          &conv_cfg, &output_tensor);
      }
      // Depthwise convolution invocation
      else if(op_type == "DepthwiseConv")
      {
        mli::krn::ref::conv2d_prepare_and_run<int16_t, int16_t, int16_t, mli_fx16_accu_t, mli::krn::fx_quant_specific_params, LAYOUT_HWC,  mli::CONV_DEPTHWISE>(
          &(tensor_map.find(input))->second,
          &conv_wt,
          &conv_bias,
          &conv_cfg, &output_tensor);
      }
      output_tensor.shape[3] = 1;
      std::cout << "\nOutput key in tensor map: " << output_name;
      tensor_map.insert(std::pair<std::string, mli_tensor>(output_name, output_tensor)); // Store the output tensor to tensor map
    }
    else if (op_type =="Add")
    {
      mli_tensor output_tensor;
      auto input = g_n.get_inputs();
      std::cout << "\nInput node 1: " << input[0];
      std::cout << "\nInput node 2: " << input[1];
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
      std::cout << "\nOutput: " << g_n.get_outputs()[0];
      tensor_map.insert(std::pair<std::string, mli_tensor>(g_n.get_outputs()[0], output_tensor));
    }
    else if (op_type =="GlobalAveragePool")
    {
      mli_tensor output_tensor;
      auto input = g_n.get_inputs();
      std::cout << "\nInput node : " << input[0];
      auto shape = (tensor_map.find(input[0]))->second.shape;
      int buf_size = 1;
      int rank = (tensor_map.find(input[0]))->second.rank;
      int start = 0, end = rank - 1;
      for (int i = 0; i < rank; i++)
      {
        buf_size = buf_size * shape[i];
      }

      create_mwnn_tensor_output(&output_tensor, buf_size);

      mli_pool_cfg pool_cfg;
      pool_cfg.kernel_width = (tensor_map.find(input[0]))->second.shape[0];
      pool_cfg.kernel_height = (tensor_map.find(input[0]))->second.shape[1];
      pool_cfg.stride_width = 1;
      pool_cfg.stride_height = 1;
      pool_cfg.padding_top = 0;
      pool_cfg.padding_bottom = 0;
      pool_cfg.padding_left = 0;
      pool_cfg.padding_right = 0;

      mli::krn::mli_krn_avepool_hwc<int16_t, mli_fx16_accu_t, 0>(&(tensor_map.find(input[0]))->second, &pool_cfg, &output_tensor);

      tensor_map.insert(std::pair<std::string, mli_tensor>(g_n.get_outputs()[0], output_tensor));
    }
    else if (op_type =="Reshape")
    {
      auto input = g_n.get_inputs();
      (tensor_map.find(input[0]))->second.shape[2] = 1;
      tensor_map.insert(std::pair<std::string, mli_tensor>(g_n.get_outputs()[0], (tensor_map.find(input[0]))->second));
    }
    else if (op_type =="Softmax")
    {
      auto input = g_n.get_inputs();
      tensor_map.insert(std::pair<std::string, mli_tensor>(g_n.get_outputs()[0], (tensor_map.find(input[0]))->second));
    }
  }
}

} //namespace metawarenn
