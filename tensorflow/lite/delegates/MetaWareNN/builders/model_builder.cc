#include "model_builder.h"

namespace tflite {
namespace delegates {
namespace metawarenn {

ModelBuilder::ModelBuilder(std::vector<int> nodes)
    : subgraph_nodes_(nodes) {}

std::shared_ptr<::metawarenn::MWNNGraph> ModelBuilder::BuildGraph(TfLiteContext* context, std::string subgraph_name) {
  std::cout<<"\nBuildGraph!!"<<std::endl;

  /*Create MetaWareNN High Level Graph Representation from TFLite SubGraph Nodes*/
  std::shared_ptr<::metawarenn::MWNNGraph> mwnn_graph_ptr = std::make_shared<::metawarenn::MWNNGraph>();
  mwnn_graph_ptr->set_name(subgraph_name);

  std::cout << "\n----------------------------------------------------------------------------------------------------------------\n";
  std::cout << "\n MWNN Graph Name : " << mwnn_graph_ptr->get_name() << " with size as " << subgraph_nodes_.size() << " nodes";

  TfLiteNode* node;
  TfLiteRegistration* reg;

  //Set Graph Input Node
  context->GetNodeAndRegistration(context, subgraph_nodes_[0], &node, &reg);
  int tensor_id = node->inputs->data[0];
  const auto& input_tensor = context->tensors[tensor_id];
  std::vector<int> dims_ip_vec(input_tensor.dims->data, input_tensor.dims->data + input_tensor.dims->size);
  ::metawarenn::MWNNValueInfo mwnn_input(input_tensor.name, dims_ip_vec, get_mwnn_type_tf(input_tensor.type));
  mwnn_graph_ptr->set_graph_ip_names(input_tensor.name);
  auto ip_node = mwnn_input.get_node();
  mwnn_graph_ptr->mwnn_graph_nodes[mwnn_input.get_name()] = std::move(ip_node);
  //Fills Graph Input Tensor Details - Name, Dims
  ::metawarenn::MWNNTensor mwnn_ip_tensor(mwnn_input.get_name(), mwnn_input.get_type(), mwnn_input.get_dims());
  mwnn_graph_ptr->set_graph_ip_tensor(mwnn_ip_tensor);

  //Set Graph Output Node
  context->GetNodeAndRegistration(context, subgraph_nodes_[subgraph_nodes_.size()-1], &node, &reg);
  tensor_id = node->outputs->data[0];
  const auto& output_tensor = context->tensors[tensor_id];
  std::vector<int> dims_op_vec(output_tensor.dims->data, output_tensor.dims->data + output_tensor.dims->size);
  ::metawarenn::MWNNValueInfo mwnn_output(output_tensor.name, dims_op_vec, get_mwnn_type_tf(output_tensor.type));
  mwnn_graph_ptr->set_graph_ip_names(output_tensor.name);
  //Fills Graph Output Tensor Details - Name, Dims
  ::metawarenn::MWNNTensor mwnn_op_tensor(mwnn_output.get_name(), mwnn_output.get_type(), mwnn_output.get_dims());
  mwnn_graph_ptr->set_graph_op_tensor(mwnn_op_tensor);

  for (size_t node_index = 0; node_index < subgraph_nodes_.size(); node_index++) {
    std::cout << "\n -------------------------------------------------------------------------------------------------------------";
    TfLiteNode* node;
    TfLiteRegistration* reg;
    const auto status = context->GetNodeAndRegistration(context, subgraph_nodes_[node_index], &node, &reg);
    auto op_type = reg->builtin_code;

    std::string node_name;
    std::string node_op_type;
    std::vector<std::string> node_inputs;
    std::vector<std::string> node_outputs;
    std::vector<::metawarenn::MWNNAttribute> node_attributes;

    //Op Names are added to follow the same pattern like in ONNX as of now.
    if (op_type == kTfLiteBuiltinConv2d) {
      node_op_type = "Conv";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteConvParams* conv_params = reinterpret_cast<const TfLiteConvParams*>(node->builtin_data);
      const int weight_tensor_id = node->inputs->data[1];
      const auto& weight_tensor = context->tensors[weight_tensor_id];

      ::metawarenn::MWNNAttribute mwnn_attr_dilate("dilations", std::vector<int>{conv_params->dilation_height_factor, conv_params->dilation_width_factor});
      node_attributes.emplace_back(mwnn_attr_dilate);
      ::metawarenn::MWNNAttribute mwnn_attr_stride("strides", std::vector<int>{conv_params->stride_height, conv_params->stride_width});
      node_attributes.emplace_back(mwnn_attr_stride);
      ::metawarenn::MWNNAttribute mwnn_attr_kernel_shape("kernel_shape", std::vector<int>{weight_tensor.dims->data[1], weight_tensor.dims->data[2]});
      node_attributes.emplace_back(mwnn_attr_kernel_shape);

      int activation_type;
      if(conv_params->activation == ::tflite::ActivationFunctionType_NONE)
        activation_type = ::metawarenn::ActivationType::Activation_None;
      else if(conv_params->activation == ::tflite::ActivationFunctionType_RELU)
        activation_type = ::metawarenn::ActivationType::Activation_Relu;
      else if(conv_params->activation == ::tflite::ActivationFunctionType_RELU6)
        activation_type = ::metawarenn::ActivationType::Activation_Relu6;

      ::metawarenn::MWNNAttribute mwnn_attr_activation("activation", std::vector<int>{activation_type});
      node_attributes.emplace_back(mwnn_attr_activation);

      if(conv_params->padding == kTfLitePaddingSame) {
        const int input_tensor_id = node->inputs->data[0];
        const auto& input_tensor = context->tensors[input_tensor_id];

        int in_height = input_tensor.dims->data[1];
        int in_width = input_tensor.dims->data[2];
        int filter_height = weight_tensor.dims->data[1];
        int filter_width = weight_tensor.dims->data[2];
        int total_height_pad, total_width_pad;
        int pad_top, pad_bottom, pad_left, pad_right;

        if((in_height%conv_params->stride_height) == 0)
          total_height_pad = std::max((filter_height - conv_params->stride_height), 0);
        else
          total_height_pad = std::max((filter_height - (in_height%conv_params->stride_height)), 0);

        if((in_width%conv_params->stride_width) == 0)
          total_width_pad = std::max((filter_width - conv_params->stride_width), 0);
        else
          total_width_pad = std::max((filter_width - (in_width%conv_params->stride_width)), 0);

        pad_top = floor(total_height_pad / 2);
        pad_bottom = total_height_pad - pad_top;
        pad_left = floor(total_width_pad / 2);
        pad_right = total_width_pad - pad_left;

        ::metawarenn::MWNNAttribute mwnn_attr_pad("pads", std::vector<int>{pad_top, pad_left, pad_bottom, pad_right});
        node_attributes.emplace_back(mwnn_attr_pad);
      }
      else {
        ::metawarenn::MWNNAttribute mwnn_attr_pad("pads", std::vector<int>{0, 0, 0, 0});
        node_attributes.emplace_back(mwnn_attr_pad);
      }
    }
    else if (op_type == kTfLiteBuiltinDepthwiseConv2d) {
      node_op_type = "DepthwiseConv";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteDepthwiseConvParams* depthwise_conv_params = reinterpret_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);
      const int weight_tensor_id = node->inputs->data[1];
      const auto& weight_tensor = context->tensors[weight_tensor_id];

      ::metawarenn::MWNNAttribute mwnn_attr_dilate("dilations", std::vector<int>{depthwise_conv_params->dilation_height_factor, depthwise_conv_params->dilation_width_factor});
      node_attributes.emplace_back(mwnn_attr_dilate);
      ::metawarenn::MWNNAttribute mwnn_attr_stride("strides", std::vector<int>{depthwise_conv_params->stride_height, depthwise_conv_params->stride_width});
      node_attributes.emplace_back(mwnn_attr_stride);
      ::metawarenn::MWNNAttribute mwnn_attr_kernel_shape("kernel_shape", std::vector<int>{weight_tensor.dims->data[1], weight_tensor.dims->data[2]});
      node_attributes.emplace_back(mwnn_attr_kernel_shape);
      ::metawarenn::MWNNAttribute mwnn_attr_group("group", std::vector<int>{weight_tensor.dims->data[3]});
      node_attributes.emplace_back(mwnn_attr_group);
      int activation_type;
      if(depthwise_conv_params->activation == ::tflite::ActivationFunctionType_NONE)
        activation_type = ::metawarenn::ActivationType::Activation_None;
      else if(depthwise_conv_params->activation == ::tflite::ActivationFunctionType_RELU)
        activation_type = ::metawarenn::ActivationType::Activation_Relu;
      else if(depthwise_conv_params->activation == ::tflite::ActivationFunctionType_RELU6)
        activation_type = ::metawarenn::ActivationType::Activation_Relu6;

      ::metawarenn::MWNNAttribute mwnn_attr_activation("activation", std::vector<int>{activation_type});
      node_attributes.emplace_back(mwnn_attr_activation);

      if(depthwise_conv_params->padding == kTfLitePaddingSame) {
        const int input_tensor_id = node->inputs->data[0];
        const auto& input_tensor = context->tensors[input_tensor_id];
        const int weight_tensor_id = node->inputs->data[1];
        const auto& weight_tensor = context->tensors[weight_tensor_id];

        int in_height = input_tensor.dims->data[1];
        int in_width = input_tensor.dims->data[2];
        int filter_height = weight_tensor.dims->data[1];
        int filter_width = weight_tensor.dims->data[2];
        int total_height_pad, total_width_pad;
        int pad_top, pad_bottom, pad_left, pad_right;

        if((in_height%depthwise_conv_params->stride_height) == 0)
          total_height_pad = std::max((filter_height - depthwise_conv_params->stride_height), 0);
        else
          total_height_pad = std::max((filter_height - (in_height%depthwise_conv_params->stride_height)), 0);

        if((in_width%depthwise_conv_params->stride_width) == 0)
          total_width_pad = std::max((filter_width - depthwise_conv_params->stride_width), 0);
        else
          total_width_pad = std::max((filter_width - (in_width%depthwise_conv_params->stride_width)), 0);

        pad_top = floor(total_height_pad / 2);
        pad_bottom = total_height_pad - pad_top;
        pad_left = floor(total_width_pad / 2);
        pad_right = total_width_pad - pad_left;

        ::metawarenn::MWNNAttribute mwnn_attr_pad("pads",std::vector<int> {pad_top, pad_left, pad_bottom, pad_right});
        node_attributes.emplace_back(mwnn_attr_pad);
      }
      else {
        ::metawarenn::MWNNAttribute mwnn_attr_pad("pads", std::vector<int>{0, 0, 0, 0});
        node_attributes.emplace_back(mwnn_attr_pad);
      }
    }
    else if (op_type == kTfLiteBuiltinAdd) {
      node_op_type = "Add";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
    }
    else if (op_type == kTfLiteBuiltinRelu) {
      node_op_type = "Relu";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
    }
    else if (op_type == kTfLiteBuiltinMaxPool2d || op_type == kTfLiteBuiltinAveragePool2d) {
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLitePoolParams* pool_params = reinterpret_cast<const TfLitePoolParams*>(node->builtin_data);
      if(op_type == kTfLiteBuiltinAveragePool2d) {
        const int input_tensor_id = node->inputs->data[0];
        const auto& input_tensor = context->tensors[input_tensor_id];
        int in_height = input_tensor.dims->data[1];
        int in_width = input_tensor.dims->data[2];
        if(in_height == pool_params->filter_height && in_width == pool_params->filter_width)
          node_op_type = "GlobalAveragePool";
        else
          node_op_type = "AveragePool";
      }
      else
        node_op_type = "MaxPool";

      ::metawarenn::MWNNAttribute mwnn_attr_stride("strides", std::vector<int>{pool_params->stride_height, pool_params->stride_width});
      node_attributes.emplace_back(mwnn_attr_stride);
      ::metawarenn::MWNNAttribute mwnn_attr_kernel_shape("kernel_shape", std::vector<int>{pool_params->filter_height, pool_params->filter_width});
      node_attributes.emplace_back(mwnn_attr_kernel_shape);
      ::metawarenn::MWNNAttribute mwnn_attr_activation("activation", std::vector<int>{pool_params->activation});
      node_attributes.emplace_back(mwnn_attr_activation);
      if(pool_params->padding == kTfLitePaddingSame) {
        const int input_tensor_id = node->inputs->data[0];
        const auto& input_tensor = context->tensors[input_tensor_id];

        int in_height = input_tensor.dims->data[1];
        int in_width = input_tensor.dims->data[2];
        int filter_height = pool_params->filter_height;
        int filter_width = pool_params->filter_width;
        int total_height_pad, total_width_pad;
        int pad_top, pad_bottom, pad_left, pad_right;

        if((in_height%pool_params->stride_height) == 0)
          total_height_pad = std::max((filter_height - pool_params->stride_height), 0);
        else
          total_height_pad = std::max((filter_height - (in_height%pool_params->stride_height)), 0);

        if((in_width%pool_params->stride_width) == 0)
          total_width_pad = std::max((filter_width - pool_params->stride_width), 0);
        else
          total_width_pad = std::max((filter_width - (in_width%pool_params->stride_width)), 0);

        pad_top = floor(total_height_pad / 2);
        pad_bottom = total_height_pad - pad_top;
        pad_left = floor(total_width_pad / 2);
        pad_right = total_width_pad - pad_left;

        ::metawarenn::MWNNAttribute mwnn_attr_pad("pads", std::vector<int>{pad_top, pad_left, pad_bottom, pad_right});
        node_attributes.emplace_back(mwnn_attr_pad);
      }
      else {
        ::metawarenn::MWNNAttribute mwnn_attr_pad("pads", std::vector<int>{0, 0, 0, 0});
        node_attributes.emplace_back(mwnn_attr_pad);
      }
    }
    else if (op_type == kTfLiteBuiltinMul) {
      node_op_type = "Mul";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteMulParams* mul_params = reinterpret_cast<const TfLiteMulParams*>(node->builtin_data);
      ::metawarenn::MWNNAttribute mwnn_attr_activation("activation", std::vector<int>{mul_params->activation});
      node_attributes.emplace_back(mwnn_attr_activation);
    }
    else if (op_type == kTfLiteBuiltinConcatenation) {
      node_op_type = "Concat";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteConcatenationParams* concat_params = reinterpret_cast<const TfLiteConcatenationParams*>(node->builtin_data);
      ::metawarenn::MWNNAttribute mwnn_attr_activation("activation", std::vector<int>{concat_params->activation});
      node_attributes.emplace_back(mwnn_attr_activation);
      ::metawarenn::MWNNAttribute mwnn_attr_axis("axis", std::vector<int>{concat_params->axis});
      node_attributes.emplace_back(mwnn_attr_axis);
    }
    else if (op_type == kTfLiteBuiltinMean) {
      node_op_type = "Mean";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
    }
    else if (op_type == kTfLiteBuiltinFullyConnected) {
      node_op_type = "FullyConnected";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteFullyConnectedParams* fc_params = reinterpret_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);
      ::metawarenn::MWNNAttribute mwnn_attr_asymmetric_quantize_inputs("asymmetric_quantize_inputs",std::vector<int> {fc_params->asymmetric_quantize_inputs});
      node_attributes.emplace_back(mwnn_attr_asymmetric_quantize_inputs);
      ::metawarenn::MWNNAttribute mwnn_attr_keep_num_dims("keep_num_dims", std::vector<int>{fc_params->keep_num_dims});
      node_attributes.emplace_back(mwnn_attr_keep_num_dims);
      ::metawarenn::MWNNAttribute mwnn_attr_activation("activation", std::vector<int>{fc_params->activation});
      node_attributes.emplace_back(mwnn_attr_activation);
      ::metawarenn::MWNNAttribute mwnn_attr_weights_format("weights_format", std::vector<int>{fc_params->weights_format});
      node_attributes.emplace_back(mwnn_attr_weights_format);
    }
    else if (op_type == kTfLiteBuiltinSplit) {
      node_op_type = "Split";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteSplitParams* split_params = reinterpret_cast<const TfLiteSplitParams*>(node->builtin_data);
      ::metawarenn::MWNNAttribute mwnn_attr_num_splits("num_splits  ", std::vector<int>{split_params->num_splits});
      node_attributes.emplace_back(mwnn_attr_num_splits);
    }
    else if (op_type == kTfLiteBuiltinPad) {
      node_op_type = "Pad";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
    }
    else if (op_type == kTfLiteBuiltinStridedSlice) {
      node_op_type = "StridedSlice";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteStridedSliceParams* strided_slice_params = reinterpret_cast<const TfLiteStridedSliceParams*>(node->builtin_data);
      ::metawarenn::MWNNAttribute mwnn_attr_begin_mask("begin_mask", std::vector<int>{strided_slice_params->begin_mask});
      node_attributes.emplace_back(mwnn_attr_begin_mask);
      ::metawarenn::MWNNAttribute mwnn_attr_ellipsis_mask("ellipsis_mask", std::vector<int>{strided_slice_params->ellipsis_mask});
      node_attributes.emplace_back(mwnn_attr_ellipsis_mask);
      ::metawarenn::MWNNAttribute mwnn_attr_end_mask("end_mask", std::vector<int>{strided_slice_params->end_mask});
      node_attributes.emplace_back(mwnn_attr_end_mask);
      ::metawarenn::MWNNAttribute mwnn_attr_new_axis_mask("new_axis_mask", std::vector<int>{strided_slice_params->new_axis_mask});
      node_attributes.emplace_back(mwnn_attr_new_axis_mask);
      ::metawarenn::MWNNAttribute mwnn_attr_shrink_axis_mask("shrink_axis_mask", std::vector<int>{strided_slice_params->shrink_axis_mask});
      node_attributes.emplace_back(mwnn_attr_shrink_axis_mask);
    }
    else if (op_type == kTfLiteBuiltinSqueeze) {
      node_op_type = "Squeeze";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteSqueezeParams* squeeze_params = reinterpret_cast<const TfLiteSqueezeParams*>(node->builtin_data);
      ::metawarenn::MWNNAttribute mwnn_attr_num_squeeze_dims("num_squeeze_dims", std::vector<int>{squeeze_params->num_squeeze_dims});
      node_attributes.emplace_back(mwnn_attr_num_squeeze_dims);
      std::vector<int> squeeze_dims(squeeze_params->squeeze_dims, squeeze_params->squeeze_dims + squeeze_params->num_squeeze_dims);
      ::metawarenn::MWNNAttribute mwnn_attr_squeeze_dims("squeeze_dims", std::vector<int>{squeeze_dims});
      node_attributes.emplace_back(mwnn_attr_squeeze_dims);
    }
    else if (op_type == kTfLiteBuiltinReshape) {
      node_op_type = "Reshape";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteReshapeParams* reshape_params = reinterpret_cast<const TfLiteReshapeParams*>(node->builtin_data);
      ::metawarenn::MWNNAttribute mwnn_attr_shape("shape", std::vector<int>{reshape_params->shape[0], reshape_params->shape[1]});
      node_attributes.emplace_back(mwnn_attr_shape);
    }
    else if (op_type == kTfLiteBuiltinSoftmax) {
      node_op_type = "Softmax";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteSoftmaxParams* softmax_params = reinterpret_cast<const TfLiteSoftmaxParams*>(node->builtin_data);
      ::metawarenn::MWNNAttribute mwnn_attr_beta("beta",std::vector<int> {(int32_t)softmax_params->beta});
      node_attributes.emplace_back(mwnn_attr_beta);
    }
    else {
      std::cout << "\n Unsupported op_type: " << op_type;
      exit(1);
    }

    for (int i = 0; i < node->inputs->size; ++i) {
      const int tensor_id = node->inputs->data[i];
      node_inputs.emplace_back(context->tensors[tensor_id].name);
    }
    for (int i = 0; i < node->outputs->size; ++i) {
      const int tensor_id = node->outputs->data[i];
      node_outputs.emplace_back(context->tensors[tensor_id].name);
    }
    for (int i = 0; i < node->inputs->size; ++i) {
      const int tensor_id = node->inputs->data[i];
      const auto& input_tensor = context->tensors[tensor_id];

      if (input_tensor.allocation_type == kTfLiteMmapRo) {
          std::vector<int> dims_vec(input_tensor.dims->data, input_tensor.dims->data + input_tensor.dims->size);
          auto num_tensor_elements = std::accumulate(begin(dims_vec), end(dims_vec), 1, std::multiplies<int>());
          std::vector<float> tensor_vec(input_tensor.data.f, input_tensor.data.f + num_tensor_elements);

          ::metawarenn::MWNNTensor mwnn_tensor(input_tensor.name, dims_vec, get_mwnn_type_tf(input_tensor.type), tensor_vec);
          mwnn_graph_ptr->set_graph_initializers(mwnn_tensor);

          auto const_node = mwnn_tensor.get_constant_node();
          mwnn_graph_ptr->mwnn_graph_nodes[mwnn_tensor.get_name()] = std::move(const_node);

          ::metawarenn::MWNNValueInfo mwnn_input(input_tensor.name, dims_vec, get_mwnn_type_tf(input_tensor.type));
          mwnn_graph_ptr->mwnn_initializer_names.insert(input_tensor.name);
      }
    }
    ::metawarenn::MWNNNode mwnn_node(node_name, node_op_type, node_attributes, node_inputs, node_outputs);
    mwnn_graph_ptr->set_graph_nodes(mwnn_node);
    auto op_node = mwnn_node.get_node();
    mwnn_graph_ptr->mwnn_graph_nodes[mwnn_node.get_name()] = std::move(op_node);
  }
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
  static int subgraph_counter = 0;
  subgraph_counter++;
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
        //std::cout << "\n Name : " << g_t.get_name();
        for(auto node: mwnn_graph->get_graph_nodes()) {
          if(node.get_op_type() == "Conv" && g_t.get_name() == node.get_inputs()[1]) {
            /*std::cout << "\t Dims : ";
            for (auto dim : g_t.get_dims())
              std::cout << dim << ",";*/
            ::metawarenn::optimizer::ConvertLayout cl(mwnn_graph, g_t, 0, HWC_TO_CHW, true);
            manager.register_pass(cl);
          }
        }
      }
    }
    for (auto g_t : mwnn_graph->get_graph_ip_tensor()) {
      if(g_t.get_dims().size() == 4) {
        /*std::cout << "\n Name : " << g_t.get_name();
        std::cout << "\t Dims : ";
        for (auto dim : g_t.get_dims())
          std::cout << dim << ",";*/
        ::metawarenn::optimizer::ConvertLayout cl(mwnn_graph, g_t, 0, HWC_TO_CHW, false);
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
  ::metawarenn::optimizer::CalculateOffset co(mwnn_graph);
  manager.register_pass(co);
  manager.run_passes();

  #if INVOKE_NNAC
    std::cout << "\n ---------------------------Graph----------------------------- \n";
    std::cout << "\n Graph Name : " << mwnn_graph->get_name();
    ::MWNN::MWNNGraphProto mwnn_graph_proto;
    mwnn_graph_proto.set_name(mwnn_graph->get_name());
    for (auto g_ip : mwnn_graph->get_graph_ip_names())
      mwnn_graph_proto.add_ip_name((g_ip));
    for (auto g_op : mwnn_graph->get_graph_op_names())
      mwnn_graph_proto.add_op_name((g_op));

    std::cout << "\n -----------------------Graph Inputs-------------------------- \n";
    for (auto g_ip : mwnn_graph->get_graph_ip_tensor()) {
      std::cout << "\n Input Name : " << g_ip.get_name();
      std::cout << "\n Data Type : " << g_ip.get_type();
      std::cout << "\n Input Dims : ";
      auto input = mwnn_graph_proto.add_input();
      input->set_name(g_ip.get_name());
      input->set_type(g_ip.get_type());
      for (auto dim : g_ip.get_dims()) {
        std::cout << dim << ",";
        input->add_dims(dim);
      }
    }
    std::cout << "\n -----------------------Graph Outputs-------------------------- \n";
    for (auto g_op : mwnn_graph->get_graph_op_tensor()) {
      std::cout << "\n Output Name : " << g_op.get_name();
      std::cout << "\n Data Type : " << g_op.get_type();
      std::cout << "\n Output Dims : ";
      auto output = mwnn_graph_proto.add_output();
      output->set_name(g_op.get_name());
      output->set_type(g_op.get_type());
      for (auto dim : g_op.get_dims()) {
        std::cout << dim << ",";
        output->add_dims(dim);
      }
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
      for (auto n_ip : g_n.get_inputs()) {
        std::cout << "\n Input : n_ip : " << n_ip;
        node->add_ip_name((n_ip));
      }
      for (auto n_op : g_n.get_outputs()) {
        std::cout << "\n Output : n_op : " << n_op;
        node->add_op_name((n_op));
      }
      std::cout << "\n ---------------------------------------------------------------- ";
      for (auto attribute : g_n.get_attributes()) {
        std::cout << "\n Attribute Name : " << attribute.get_name();
        std::cout << "\n Attribute Data : ";
        auto attr = node->add_attribute();
        attr->set_name(attribute.get_name());
        attr->set_type(attribute.get_type());
        if(attribute.get_type() == 6) { //int data
          for(int i = 0; i < attribute.get_int_data().size(); i++){
            attr->add_int_data(attribute.get_int_data()[i]);
            std::cout << attribute.get_int_data()[i] << ",";
          }
        }
        else if(attribute.get_type() == 3) { //float data
          for(int i = 0; i < attribute.get_float_data().size(); i++){
            attr->add_float_data(attribute.get_float_data()[i]);
            std::cout << attribute.get_float_data()[i] << ",";
          }
        }
        else if(attribute.get_type() == 12) { //string data
          for(int i = 0; i < attribute.get_string_data().size(); i++){
            attr->add_string_data(attribute.get_string_data()[i]);
            std::cout << attribute.get_string_data()[i] << ",";
          }
        }
      }
    }
    std::cout << "\n -----------------------Graph Tensors-------------------------- \n";
    for (auto g_t : mwnn_graph->get_graph_initializers()) {
      auto initializer = mwnn_graph_proto.add_initializer();
      initializer->set_name(g_t.get_name());
      initializer->set_type(g_t.get_type());
      std::cout << "\n Name : " << g_t.get_name();
      std::cout << "\n Type : " << g_t.get_type();
      std::cout << "\n Dims : ";
      for (auto dim : g_t.get_dims()) {
        std::cout << dim << ",";
        initializer->add_dims(dim);
      }
      //std::cout << "\n Tensor values : ";
      for (auto t_val : g_t.get_tensor()) {
        //std::cout << t_val << ",";
        initializer->add_float_data(t_val);
      }
    }

    std::cout << "\n Graph Name : " << mwnn_graph->get_name();
    std::string name = mwnn_graph->get_name();
    char* mwnn_op_path = nullptr;
    mwnn_op_path = getenv("NNAC_DUMPS_PATH");
    if(!IsPathExist(std::string(mwnn_op_path))) {
      int check = mkdir(mwnn_op_path, 0777);
      if(check != 0) {
        std::cout << "\nPlease check the directory path to store the serialized binary!!!!!";
        exit(1);
      }
    }
    auto mwnn_proto_bin = std::string(mwnn_op_path) + std::string(name) + ".bin";

    int fp = open(mwnn_proto_bin.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    std::cout << fp;
    std::cout << mwnn_graph_proto.SerializeToFileDescriptor(fp);
    close(fp);

    char* mwnn_lib_path = nullptr;
    mwnn_lib_path = getenv("METAWARENN_LIB_PATH");
    if(!IsPathExist(std::string(mwnn_lib_path)))
      std::cout << "\nPlease check the MetaWareNN Library path!!!";
    std::cout << "\n\n=================Initiating NNAC python script via shell script======================\n";
    std::string cmd = "bash " + std::string(mwnn_lib_path) +"/mwnnconvert/mwnn_convert.sh " + mwnn_proto_bin + " " + mwnn_op_path + " " + name + " " + std::to_string(subgraph_counter);
    const char *command = cmd.c_str();
    system(command);
  #endif

  return kTfLiteOk;
  }
} // namespace metawarenn
} // namespace delegates
} // namespace tflite
