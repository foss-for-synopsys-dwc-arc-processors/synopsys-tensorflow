#include "metawarenn_graph.h"

namespace metawarenn {

//ONNXConstructor
#if ONNX
MWNNGraph::MWNNGraph(GraphProto& onnx_graph_proto, std::string graph_name) {
  name = graph_name;

  for (auto tensor_proto : onnx_graph_proto.initializer()) {
    MWNNTensor mwnn_tensor(tensor_proto);
    mwnn_initializer_tensors.emplace_back(mwnn_tensor);
    mwnn_initializer_names.insert(mwnn_tensor.get_name());
    auto const_node = mwnn_tensor.get_constant_node();
    mwnn_graph_nodes[mwnn_tensor.get_name()] = std::move(const_node);
  }
  for (auto node_proto : onnx_graph_proto.node()) {
    MWNNNode mwnn_node(node_proto);
    mwnn_nodes.emplace_back(mwnn_node);
    auto node = mwnn_node.get_node();
    mwnn_graph_nodes[mwnn_node.get_name()] = std::move(node);
  }
  for (auto ip_value_info_proto : onnx_graph_proto.input()) {
    MWNNValueInfo mwnn_input(ip_value_info_proto);
    if(mwnn_initializer_names.count(mwnn_input.get_name()))
      continue;
    else {
      std::string ip_name = mwnn_input.get_name();
      mwnn_graph_ip_names.emplace_back(ip_name);
      auto ip_node = mwnn_input.get_node();
      mwnn_graph_nodes[ip_name] = std::move(ip_node);
      //Fills Graph Input Tensor Details - Name, Dims
      MWNNTensor mwnn_ip_tensor(mwnn_input.get_name(), mwnn_input.get_type(), mwnn_input.get_dims());
      mwnn_graph_ip_tensors.emplace_back(mwnn_ip_tensor);
    }
  }
  for (auto op_value_info_proto : onnx_graph_proto.output()) {
    MWNNValueInfo mwnn_output(op_value_info_proto);
    mwnn_graph_op_names.emplace_back(mwnn_output.get_name());
    //Fills Graph Output Tensor Details - Name, Dims
    MWNNTensor mwnn_op_tensor(mwnn_output.get_name(), mwnn_output.get_type(), mwnn_output.get_dims());
    mwnn_graph_op_tensors.emplace_back(mwnn_op_tensor);
  }
}
#endif

#if TFLITE
//TFConstructor
MWNNGraph::MWNNGraph(TfLiteContext* context, std::vector<int> subgraph_nodes_, std::string subgraph_name) {
  name = subgraph_name;
  std::cout << "\n----------------------------------------------------------------------------------------------------------------\n";
  std::cout << "\n MWNN Graph Name : " << get_name() << " with size as " << subgraph_nodes_.size() << " nodes";

  TfLiteNode* node;
  TfLiteRegistration* reg;

  //Set Graph Input Node
  context->GetNodeAndRegistration(context, subgraph_nodes_[0], &node, &reg);
  int tensor_id = node->inputs->data[0];
  const auto& input_tensor = context->tensors[tensor_id];
  std::vector<int> dims_ip_vec(input_tensor.dims->data, input_tensor.dims->data + input_tensor.dims->size);
  ::metawarenn::MWNNValueInfo mwnn_input(input_tensor.name, dims_ip_vec, input_tensor.type);
  mwnn_graph_ip_names.emplace_back(input_tensor.name);
  auto ip_node = mwnn_input.get_node();
  mwnn_graph_nodes[mwnn_input.get_name()] = std::move(ip_node);
  //Fills Graph Input Tensor Details - Name, Dims
  MWNNTensor mwnn_ip_tensor(mwnn_input.get_name(), mwnn_input.get_type(), mwnn_input.get_dims());
  mwnn_graph_ip_tensors.emplace_back(mwnn_ip_tensor);

  //Set Graph Output Node
  context->GetNodeAndRegistration(context, subgraph_nodes_[subgraph_nodes_.size()-1], &node, &reg);
  tensor_id = node->outputs->data[0];
  const auto& output_tensor = context->tensors[tensor_id];
  std::vector<int> dims_op_vec(output_tensor.dims->data, output_tensor.dims->data + output_tensor.dims->size);
  ::metawarenn::MWNNValueInfo mwnn_output(output_tensor.name, dims_op_vec, output_tensor.type);
  mwnn_graph_op_names.emplace_back(output_tensor.name);
  //Fills Graph Output Tensor Details - Name, Dims
  MWNNTensor mwnn_op_tensor(mwnn_output.get_name(), mwnn_output.get_type(), mwnn_output.get_dims());
  mwnn_graph_op_tensors.emplace_back(mwnn_op_tensor);

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
        activation_type = ActivationType::Activation_None;
      else if(conv_params->activation == ::tflite::ActivationFunctionType_RELU)
        activation_type = ActivationType::Activation_Relu;
      else if(conv_params->activation == ::tflite::ActivationFunctionType_RELU6)
        activation_type = ActivationType::Activation_Relu6;

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
        activation_type = ActivationType::Activation_None;
      else if(depthwise_conv_params->activation == ::tflite::ActivationFunctionType_RELU)
        activation_type = ActivationType::Activation_Relu;
      else if(depthwise_conv_params->activation == ::tflite::ActivationFunctionType_RELU6)
        activation_type = ActivationType::Activation_Relu6;

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

          ::metawarenn::MWNNTensor mwnn_tensor(input_tensor.name, dims_vec, input_tensor.type, tensor_vec);
          mwnn_initializer_tensors.emplace_back(mwnn_tensor);

          auto const_node = mwnn_tensor.get_constant_node();
          mwnn_graph_nodes[mwnn_tensor.get_name()] = std::move(const_node);

          ::metawarenn::MWNNValueInfo mwnn_input(input_tensor.name, dims_vec, input_tensor.type);
          mwnn_initializer_names.insert(input_tensor.name);
      }
    }
    ::metawarenn::MWNNNode mwnn_node(node_name, node_op_type, node_attributes, node_inputs, node_outputs);
    mwnn_nodes.emplace_back(mwnn_node);
    auto op_node = mwnn_node.get_node();
    mwnn_graph_nodes[mwnn_node.get_name()] = std::move(op_node);
  }
}
#endif

#if GLOW
//GLOWConstructor
MWNNGraph::MWNNGraph(Function *F, std::string subgraph_name) {
  name = subgraph_name;
  LOG(INFO) << "Function name: " << name;
  GraphPostOrderVisitor visitor(*F);
  auto node_list = visitor.getPostOrder();
  auto global_output_name = "";
  for (auto *node : node_list) {
      LOG(INFO) << "==============================================================================================================";
      std::string node_name;
      std::string node_op_type;
      std::vector<std::string> node_inputs;
      std::vector<std::string> node_outputs;
      std::vector<::metawarenn::MWNNAttribute> node_attributes;
      node_name = std::string(node->getName());
      auto kind = node->getKindName();
      LOG(INFO) << "Node Name: " << node_name << "\tOp type: " << node->getKindName();
      std::vector<std::string> non_op_types;
      non_op_types = {"Constant", "Placeholder", "Save"};
      if((std::count(non_op_types.begin(), non_op_types.end(), kind)) < 1)
      {
      switch (node->getKind())
      {
        case Kinded::Kind::ConvolutionNodeKind:
        {
            node_op_type = "Conv";
            auto *conv_node = llvm::cast<ConvolutionNode>(node);
            auto input_node = conv_node->getInput();
            auto input_name = input_node.generateNodeOutputName(true);
            node_inputs.emplace_back(input_name);
            LOG(INFO) << "input_name: " << input_name;
            auto filter_node_value = conv_node->getFilter();
            auto filter_name = filter_node_value.generateNodeOutputName(true);
            LOG(INFO) << "filter_name: " << filter_name;
            node_inputs.emplace_back(filter_name);
            mwnn_initializer_names.insert(filter_name);
            auto *filter_constant = llvm::dyn_cast<glow::Constant>(filter_node_value.getNode());
            glow::Tensor filter_tensor = filter_constant->getPayload().clone();
            auto type = filter_tensor.getType();
            glow::ElemKind data_type = type.getElementType();
            ShapeNHWC filterDims(filter_node_value.dims());
            size_t wt_size = filterDims.n * filterDims.h * filterDims.w * filterDims.c;
            std::vector<float> weights(wt_size);
            std::vector<int> weight_dims(4);
            weight_dims[0] = filterDims.n;
            weight_dims[1] = filterDims.h;
            weight_dims[2] = filterDims.w;
            weight_dims[3] = filterDims.c;
            auto handle = filter_tensor.getHandle<float>();
            int i = 0;
            for (auto elem : handle)
            {
                weights[i++] = elem;
            }
            metawarenn::MWNNTensor mwnn_weight_tensor(filter_name, weight_dims, data_type, weights);
            mwnn_initializer_tensors.emplace_back(mwnn_weight_tensor);
            auto bias_node_value = conv_node->getBias();
            auto bias_name = bias_node_value.generateNodeOutputName(true);
            // Check to avoid redundant constants in mwnn initializers
            if(!mwnn_initializer_names.count(bias_name))
            {
            LOG(INFO) << "bias_name: " << bias_name;
            auto *bias_constant = llvm::dyn_cast<glow::Constant>(bias_node_value.getNode());
            glow::Tensor bias_tensor = bias_constant->getPayload().clone();
            auto handle1 = bias_tensor.getHandle<float>();
            auto base1 = handle1.getElementPtr({static_cast<unsigned long>(0)});
            std::vector<float> bias(filterDims.n);
            std::vector<int> bias_dims(1);
            bias_dims[0] = filterDims.n;
            i = 0;
            for (auto elem : handle1)
            {
                bias[i++] = elem;
            }
            node_inputs.emplace_back(bias_name);
            mwnn_initializer_names.insert(bias_name);
            type = bias_tensor.getType();
            data_type = type.getElementType();
            metawarenn::MWNNTensor mwnn_bias_tensor(bias_name, bias_dims, data_type, bias);
            mwnn_initializer_tensors.emplace_back(mwnn_bias_tensor);
            }
            auto dilations = conv_node->getDilation();
            auto strides = conv_node->getStrides();
            auto pads = conv_node->getPads();
            auto group = conv_node->getGroup();
            metawarenn::MWNNAttribute mwnn_attr_dilate("dilations", std::vector<int>{int(dilations[0]), int(dilations[1])});
            node_attributes.emplace_back(mwnn_attr_dilate);
            metawarenn::MWNNAttribute mwnn_attr_stride("strides", std::vector<int>{int(strides[0]), int(strides[1])});
            node_attributes.emplace_back(mwnn_attr_stride);
            metawarenn::MWNNAttribute mwnn_attr_pad("pads", std::vector<int>{int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])});
            node_attributes.emplace_back(mwnn_attr_pad);
            metawarenn::MWNNAttribute mwnn_attr_group("group", std::vector<int>{int(group)});
            node_attributes.emplace_back(mwnn_attr_group);
            metawarenn::MWNNAttribute mwnn_attribute("activation", std::vector<int>{0});
            node_attributes.emplace_back(mwnn_attribute);
            metawarenn::MWNNAttribute mwnn_attr_kernel_shape("kernel_shape", std::vector<int>{(int)filterDims.h, (int)filterDims.w});
            node_attributes.emplace_back(mwnn_attr_kernel_shape);
            auto output_name = conv_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
            break;
        }
        case Kinded::Kind::ReluNodeKind:
        {
            node_op_type = "Relu";
            auto *relu_node = llvm::cast<ReluNode>(node);
            auto input_name = relu_node->getInput().generateNodeOutputName(true);
            node_inputs.emplace_back(input_name);
            LOG(INFO) << "input_name: " << input_name;
            auto output_name = relu_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
            break;
        }
        case Kinded::Kind::AvgPoolNodeKind:
        {
            node_op_type = "GlobalAveragePool";
            auto *avgpool_node = llvm::cast<AvgPoolNode>(node);
            auto kernels = avgpool_node->getKernels();
            auto strides = avgpool_node->getStrides();
            auto pads = avgpool_node->getPads();
            metawarenn::MWNNAttribute mwnn_attr_kernel_shape("kernel_shape", std::vector<int>{int(kernels[0]), int(kernels[1])});
            node_attributes.emplace_back(mwnn_attr_kernel_shape);
            metawarenn::MWNNAttribute mwnn_attr_stride("strides", std::vector<int>{int(strides[0]), int(strides[1])});
            node_attributes.emplace_back(mwnn_attr_stride);
            metawarenn::MWNNAttribute mwnn_attr_pads("pads", std::vector<int>{int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])});
            node_attributes.emplace_back(mwnn_attr_pads);
            auto input_name = avgpool_node->getInput().generateNodeOutputName(true);
            node_inputs.emplace_back(input_name);
            LOG(INFO) << "input_name: " << input_name;
            auto output_name = avgpool_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
            break;
        }
        case Kinded::Kind::AddNodeKind:
        {
            node_op_type = "Add";
            auto *add_node = llvm::cast<AddNode>(node);
            auto input1 = add_node->getLHS().generateNodeOutputName(true);
            LOG(INFO) << "input_name 1: " << input1;
            auto input2 = add_node->getRHS().generateNodeOutputName(true);
            LOG(INFO) << "input_name 2: " << input2;
            node_inputs.emplace_back(input1);
            node_inputs.emplace_back(input2);
            auto output_name = add_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
            break;
        }
        case Kinded::Kind::TransposeNodeKind:
        {
            node_op_type = "Transpose";
            auto *transpose_node = llvm::cast<TransposeNode>(node);
            auto input_name = transpose_node->getInput().generateNodeOutputName(true);
            node_inputs.emplace_back(input_name);
            LOG(INFO) << "input_name: " << input_name;
            auto output_name = transpose_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
            break;
        }
        case Kinded::Kind::ReshapeNodeKind:
        {
            node_op_type = "Reshape";
            auto *reshape_node = llvm::cast<ReshapeNode>(node);
            auto input_name = reshape_node->getInput().generateNodeOutputName(true);
            std::string initializer_name = std::string(node_name + "shape");
            auto dims = reshape_node->getDims();
            std::vector<float> dims_vec(dims.size());
            std::vector<int> dims_;
            dims_.push_back(dims.size());
            int i = 0;
            for(auto dim: dims){
              dims_vec[i++] = dim;
            }
            metawarenn::MWNNTensor mwnn_reshape_tensor(initializer_name, dims_, ElemKind::Int64ITy, dims_vec);
            mwnn_initializer_tensors.emplace_back(mwnn_reshape_tensor);
            node_inputs.emplace_back(input_name);
            node_inputs.emplace_back(initializer_name);
            mwnn_initializer_names.insert(initializer_name);
            LOG(INFO) << "input_name: " << input_name;
            auto output_name = reshape_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
            break;
        }
        case Kinded::Kind::LocalResponseNormalizationNodeKind:
        {
            node_op_type = "LRN";
            auto *lrn_node = llvm::cast<LocalResponseNormalizationNode>(node);
            metawarenn::MWNNAttribute mwnn_attr_alpha("alpha", std::vector<int>{int(lrn_node->getAlpha())});
            node_attributes.emplace_back(mwnn_attr_alpha);
            metawarenn::MWNNAttribute mwnn_attr_beta("beta", std::vector<int>{int(lrn_node->getBeta())});
            node_attributes.emplace_back(mwnn_attr_beta);
            metawarenn::MWNNAttribute mwnn_attr_half_window_size("half_window_size", std::vector<int>{int(lrn_node->getHalfWindowSize())});
            node_attributes.emplace_back(mwnn_attr_half_window_size);
            auto input_name = lrn_node->getInput().generateNodeOutputName(true);
            node_inputs.emplace_back(input_name);
            LOG(INFO) << "input_name: " << input_name;
            auto output_name = lrn_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
            break;
        }
        case Kinded::Kind::MaxPoolNodeKind:
        {
            node_op_type = "MaxPool";
            auto *maxpool_node = llvm::cast<MaxPoolNode>(node);
            auto kernels = maxpool_node->getKernels();
            auto strides = maxpool_node->getStrides();
            auto pads = maxpool_node->getPads();
            metawarenn::MWNNAttribute mwnn_attr_kernel_shape("kernel_shape", std::vector<int>{int(kernels[0]), int(kernels[1])});
            node_attributes.emplace_back(mwnn_attr_kernel_shape);
            metawarenn::MWNNAttribute mwnn_attr_stride("strides", std::vector<int>{int(strides[0]), int(strides[1])});
            node_attributes.emplace_back(mwnn_attr_stride);
            metawarenn::MWNNAttribute mwnn_attr_pad("pads", std::vector<int>{int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])});
            node_attributes.emplace_back(mwnn_attr_pad);
            auto input_name = maxpool_node->getInput().generateNodeOutputName(true);
            node_inputs.emplace_back(input_name);
            LOG(INFO) << "input_name: " << input_name;
            auto output_name = maxpool_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
            break;
        }
        case Kinded::Kind::GemmNodeKind:
        {
            node_op_type = "Gemm";
            auto *gemm_node = llvm::cast<GemmNode>(node);
            std::cout << "\n gemm inputs: " << gemm_node->getNumInputs();
            auto filter_node_value = gemm_node->getNthInput(1);
            auto filter_name = filter_node_value.generateNodeOutputName(true);
            mwnn_initializer_names.insert(filter_name);
            node_inputs.emplace_back(filter_name);
            auto *filter_constant = llvm::dyn_cast<glow::Constant>(filter_node_value.getNode());
            glow::Tensor filter_tensor = filter_constant->getPayload().clone();
            auto type = filter_tensor.getType();
            glow::ElemKind data_type = type.getElementType();
            ShapeNHWC filterDims(filter_node_value.dims());
            size_t wt_size = filterDims.n * filterDims.h; //n - height, h - width
            std::vector<float> weights(wt_size);
            std::vector<int> weight_dims(filter_constant->dims().vec().size());
            weight_dims[0] = filterDims.n;
            weight_dims[1] = filterDims.h;
            auto handle = filter_tensor.getHandle<float>();
            int i = 0;
            for (auto elem : handle)
            {
                weights[i++] = elem;
            }
            metawarenn::MWNNTensor mwnn_weight_tensor(filter_name, weight_dims, data_type, weights);
            mwnn_initializer_tensors.emplace_back(mwnn_weight_tensor);
            auto bias_node_value = gemm_node->getNthInput(2);
            auto bias_name = bias_node_value.generateNodeOutputName(true);
            LOG(INFO) << "bias_name: " << bias_name;
            node_inputs.emplace_back(bias_name);
            mwnn_initializer_names.insert(bias_name);
            auto *bias_constant = llvm::dyn_cast<glow::Constant>(bias_node_value.getNode());
            glow::Tensor bias_tensor = bias_constant->getPayload().clone();
            auto handle1 = bias_tensor.getHandle<float>();
            auto base1 = handle1.getElementPtr({static_cast<unsigned long>(0)});
            std::vector<float> bias(filterDims.n);
            std::vector<int> bias_dims(1);
            bias_dims[0] = filterDims.n;
            i = 0;
            for (auto elem : handle1)
            {
                bias[i++] = elem;
            }
            type = bias_tensor.getType();
            data_type = type.getElementType();
            metawarenn::MWNNTensor mwnn_bias_tensor(bias_name, bias_dims, data_type, bias);
            mwnn_initializer_tensors.emplace_back(mwnn_bias_tensor);
            metawarenn::MWNNAttribute mwnn_attr_alpha("alpha", std::vector<int>{int(gemm_node->getAlpha())});
            node_attributes.emplace_back(mwnn_attr_alpha);
            metawarenn::MWNNAttribute mwnn_attr_beta("beta", std::vector<int>{int(gemm_node->getBeta())});
            node_attributes.emplace_back(mwnn_attr_beta);
            auto input_name = gemm_node->getInputName(0);
            node_inputs.emplace_back(input_name);
            LOG(INFO) << "input_name: " << input_name;
            auto output_name = gemm_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
            break;
        }
        case Kinded::Kind::ConcatNodeKind:
        {
            node_op_type = "Concat";
            auto *concat_node = llvm::cast<ConcatNode>(node);
            for(int i = 0; i < concat_node->getInputs().size(); i++)
            {
              auto input_name = concat_node->getInputName(i);
              node_inputs.emplace_back(input_name);
              LOG(INFO) << "input_name: " << input_name;
            }
            auto output_name = concat_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
            break;
        }
        case Kinded::Kind::BatchNormalizationNodeKind:
        {
            node_op_type = "BatchNormalization";
            auto *batchnorm_node = llvm::cast<BatchNormalizationNode>(node);
            batchnorm_node->getEpsilon();
            batchnorm_node->getMomentum();
            auto bias_node_value = batchnorm_node->getBias();
            auto bias_name = bias_node_value.generateNodeOutputName(true);
            LOG(INFO) << "bias_name: " << bias_name;
            node_inputs.emplace_back(bias_name);
            mwnn_initializer_names.insert(bias_name);
            auto *bias_constant = llvm::dyn_cast<glow::Constant>(bias_node_value.getNode());
            glow::Tensor bias_tensor = bias_constant->getPayload().clone();
            auto handle1 = bias_tensor.getHandle<float>();
            auto base1 = handle1.getElementPtr({static_cast<unsigned long>(0)});
            std::vector<float> bias(bias_tensor.size());
            std::vector<int> bias_dims(1);
            bias_dims[0] = bias_tensor.dims()[0];
            int i = 0;
            for (auto elem : handle1)
            {
                bias[i++] = elem;
            }
            auto type = bias_tensor.getType();
            auto data_type = type.getElementType();
            metawarenn::MWNNTensor mwnn_bias_tensor(bias_name, bias_dims, data_type, bias);
            mwnn_initializer_tensors.emplace_back(mwnn_bias_tensor);
            metawarenn::MWNNAttribute mwnn_attr_momentum("momentum", std::vector<float>{batchnorm_node->getMomentum()});
            node_attributes.emplace_back(mwnn_attr_momentum);
            metawarenn::MWNNAttribute mwnn_attr_epsilon("epsilon", std::vector<float>{batchnorm_node->getEpsilon()});
            node_attributes.emplace_back(mwnn_attr_epsilon);
            auto input_name = batchnorm_node->getInputName(0);
            node_inputs.emplace_back(input_name);
            auto output_name = batchnorm_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
            break;
        }
        case Kinded::Kind::ChannelShuffleNodeKind:
        {
            node_op_type = "ChannelShuffle";
            auto *channel_shuffle_node = llvm::cast<ChannelShuffleNode>(node);
            metawarenn::MWNNAttribute mwnn_attr_group("group", std::vector<int>{int(channel_shuffle_node->getGroup())});
            node_attributes.emplace_back(mwnn_attr_group);
            metawarenn::MWNNAttribute mwnn_attr_kernel("kernel", std::vector<int>{int(channel_shuffle_node->getKernel())});
            node_attributes.emplace_back(mwnn_attr_kernel);
            auto input_name = channel_shuffle_node->getInputName(0);
            node_inputs.emplace_back(input_name);
            auto output_name = channel_shuffle_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
            break;
        }
        case Kinded::Kind::ClipNodeKind:
        {
            node_op_type = "Clip";
            auto *clip_node = llvm::cast<ClipNode>(node);
            clip_node->getMax();
            clip_node->getMin();
            metawarenn::MWNNAttribute mwnn_attr_max("max", std::vector<float>{(clip_node->getMax())});
            node_attributes.emplace_back(mwnn_attr_max);
            metawarenn::MWNNAttribute mwnn_attr_min("min", std::vector<float>{(clip_node->getMax())});
            node_attributes.emplace_back(mwnn_attr_min);
            auto input_name = clip_node->getInputName(0);
            node_inputs.emplace_back(input_name);
            auto output_name = clip_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::FullyConnectedNodeKind:
        {
            node_op_type = "FullyConnected";
            auto *fc_node = llvm::cast<FullyConnectedNode>(node);
            auto filter_node_value = fc_node->getWeights();
            auto filter_name = filter_node_value.generateNodeOutputName(true);
            auto *filter_constant = llvm::dyn_cast<glow::Constant>(filter_node_value.getNode());
            glow::Tensor filter_tensor = filter_constant->getPayload().clone();
            auto type = filter_tensor.getType();
            glow::ElemKind data_type = type.getElementType();
            ShapeNHWC filterDims(filter_node_value.dims());
            size_t wt_size = filterDims.n * filterDims.h; //n - height, h - width
            std::vector<float> weights(wt_size);
            std::vector<int> weight_dims(filter_constant->dims().vec().size());
            weight_dims[0] = filterDims.n;
            weight_dims[1] = filterDims.h;
            auto handle = filter_tensor.getHandle<float>();
            int i = 0;
            for (auto elem : handle)
            {
                weights[i++] = elem;
            }
            mwnn_initializer_names.insert(filter_name);
            node_inputs.emplace_back(filter_name);
            metawarenn::MWNNTensor mwnn_weight_tensor(filter_name, weight_dims, data_type, weights);
            mwnn_initializer_tensors.emplace_back(mwnn_weight_tensor);
            auto bias_node_value = fc_node->getBias();;
            auto bias_name = bias_node_value.generateNodeOutputName(true);
            LOG(INFO) << "bias_name: " << bias_name;
            node_inputs.emplace_back(bias_name);
            mwnn_initializer_names.insert(bias_name);
            auto *bias_constant = llvm::dyn_cast<glow::Constant>(bias_node_value.getNode());
            glow::Tensor bias_tensor = bias_constant->getPayload().clone();
            auto handle1 = bias_tensor.getHandle<float>();
            auto base1 = handle1.getElementPtr({static_cast<unsigned long>(0)});
            std::vector<float> bias(filterDims.n);
            std::vector<int> bias_dims(1);
            bias_dims[0] = filterDims.n;
            i = 0;
            for (auto elem : handle1)
            {
                bias[i++] = elem;
            }
            type = bias_tensor.getType();
            data_type = type.getElementType();
            metawarenn::MWNNTensor mwnn_bias_tensor(bias_name, bias_dims, data_type, bias);
            mwnn_initializer_tensors.emplace_back(mwnn_bias_tensor);
            auto input_name = fc_node->getInputName(0);
            node_inputs.emplace_back(input_name);
            LOG(INFO) << "input_name: " << input_name;
            auto output_name = fc_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
          break;
        }
        case Kinded::Kind::SoftMaxNodeKind:
        {
            node_op_type = "Softmax";
            auto *softmax_node = llvm::cast<SoftMaxNode>(node);
            auto input_name = softmax_node->getInputName(0);
            node_inputs.emplace_back(input_name);
            LOG(INFO) << "input_name: " << input_name;
            auto output_name = softmax_node->getResult().generateNodeOutputName(true);
            node_outputs.emplace_back(output_name);
            LOG(INFO) << "output_name: " << output_name;
            break;
        }
        default:
            break;
        }
        metawarenn::MWNNNode mwnn_node(node_name, node_op_type, node_attributes, node_inputs, node_outputs);
        mwnn_nodes.emplace_back(mwnn_node);
        auto op_node = mwnn_node.get_node();
        mwnn_graph_nodes[mwnn_node.get_name()] = std::move(op_node);
        global_output_name = node_outputs[0].c_str();
      }
  }
  // Graph input and output handling
  auto &nodes = F->getNodes();
  auto &first_node = nodes.front();
  auto &last_node = nodes.back();
  auto input_name = std::string(first_node.getNthInput(0).getNode()->getName());
  auto output_name = std::string(last_node.getNthResult(0).getNode()->getName());
  for (auto &V : F->getParent()->getPlaceholders()) {
    if (!usedInFunction(V, F)) {
      continue;
    }

    auto glow_dims = V->getType()->dims();
    auto data_type = V->getType()->getElementType();
    int size = glow_dims.size();
    std::vector<int> dims(size);
    // Input dims from NCHW to NHWC
    dims[1] = int(glow_dims[3]);
    dims[3] = int(glow_dims[1]);
    dims[2] = int(glow_dims[2]);
    dims[0] = int(glow_dims[0]);
    if (getOutputSave(F, V)) {
      metawarenn::MWNNValueInfo mwnn_output(global_output_name, dims, data_type);
      mwnn_graph_op_names.emplace_back(global_output_name);
      //Fills Graph Output Tensor Details - Name, Dims
      MWNNTensor mwnn_op_tensor(mwnn_output.get_name(), mwnn_output.get_type(), mwnn_output.get_dims());
      mwnn_graph_op_tensors.emplace_back(mwnn_op_tensor);
    }
    else if(V->getName().equals(input_name)) {
      metawarenn::MWNNValueInfo mwnn_input(V->getName(), dims, data_type);
      mwnn_graph_ip_names.emplace_back(V->getName());
      //Fills Graph Input Tensor Details - Name, Dims
      MWNNTensor mwnn_ip_tensor(mwnn_input.get_name(), mwnn_input.get_type(), mwnn_input.get_dims());
      mwnn_graph_ip_tensors.emplace_back(mwnn_ip_tensor);
    }
  }
}
#endif

//TVMConstructor
#if TVM
MWNNGraph::MWNNGraph(std::vector<JSONGraphNode> graph_nodes_, std::string graph_name) {
  std::cout << "\n In TVM Metawarenn graph constructor!!  " << graph_name;
  name = graph_name;
  int layer_count = 0;
  std::vector<std::vector<int64_t>> op_shape;
  std::vector<DLDataType> dtypes;
  std::string op_name;
  for (int id = 0; id < graph_nodes_.size(); id++) {
    const auto& node = graph_nodes_[id];
    //std::cout << "\n Node Op Type : " << node.GetOpType() << " Name : " << node.GetOpName();
    if (node.GetOpType() == "kernel") {
      std::string node_name;
      std::string node_op_type;
      std::vector<std::string> node_inputs;
      std::vector<std::string> node_outputs;
      std::vector<MWNNAttribute> node_attributes;
      int out_index = 1;

      //Node Inputs Parsing
      for (size_t i = 0; i < node.GetInputs().size(); ++i) {
        auto in_node = node.GetInputs()[i];
        if(in_node.id_ > out_index)
          out_index = in_node.id_;
        std::string ip_name = "node_" + std::to_string(in_node.id_);
        node_inputs.emplace_back(ip_name);
      }
      //Node Output Parsing
      op_name = "node_" + std::to_string(out_index+1);
      node_outputs.emplace_back(op_name);

      //Node Output Shape & Type Parsing
      op_shape = node.GetOpShape();
      dtypes = node.GetOpDataType();
      //Setting MetaWareNN Op Type & Node Name
      if (node.GetOpName() == "nn.conv2d") {
        node_op_type = "Conv";
        node_name = node_op_type + std::to_string(layer_count++);
        std::vector<std::string> strides = node.GetAttr<std::vector<std::string>>("strides");
        std::vector<std::string> pads = node.GetAttr<std::vector<std::string>>("padding");
        std::vector<std::string> dilations = node.GetAttr<std::vector<std::string>>("dilation");
        int group = std::stoi(node.GetAttr<std::vector<std::string>>("groups")[0]);
        auto weight_entry = node.GetInputs()[1];
        std::vector<long int> kernel_shape = graph_nodes_[weight_entry.id_].GetOpShape()[weight_entry.index_];

        MWNNAttribute mwnn_attr_dilate("dilations", std::vector<int>({std::stoi(dilations[0]), std::stoi(dilations[1])}));
        node_attributes.emplace_back(mwnn_attr_dilate);
        MWNNAttribute mwnn_attr_stride("strides", std::vector<int>({std::stoi(strides[0]), std::stoi(strides[1])}));
        node_attributes.emplace_back(mwnn_attr_stride);
        MWNNAttribute mwnn_attr_pad("pads", std::vector<int>({std::stoi(pads[0]), std::stoi(pads[1]), std::stoi(pads[2]), std::stoi(pads[3])}));
        node_attributes.emplace_back(mwnn_attr_pad);
        MWNNAttribute mwnn_attr_group("group", std::vector<int>({group}));
        node_attributes.emplace_back(mwnn_attr_group);
        MWNNAttribute mwnn_attribute("activation", std::vector<int>({0}));
        node_attributes.emplace_back(mwnn_attribute);
        MWNNAttribute mwnn_attr_kernel_shape("kernel_shape", std::vector<int>({kernel_shape[2], kernel_shape[3]}));
        node_attributes.emplace_back(mwnn_attr_kernel_shape);
      }
      else if (node.GetOpName() == "nn.batch_norm") {
        node_op_type = "BatchNormalization";
        node_name = node_op_type + std::to_string(layer_count++);

        float epsilon = std::stof(node.GetAttr<std::vector<std::string>>("epsilon")[0]);
        MWNNAttribute mwnn_attr_epsilon("epsilon", std::vector<float>({epsilon}));
        node_attributes.emplace_back(mwnn_attr_epsilon);
      }
      else if (node.GetOpName() == "nn.relu") {
        node_op_type = "Relu";
        node_name = node_op_type + std::to_string(layer_count++);
      }
      else if (node.GetOpName() == "add") {
        node_op_type = "Add";
        node_name = node_op_type + std::to_string(layer_count++);
      }
      else if (node.GetOpName() == "nn.global_avg_pool2d") {
        node_op_type = "GlobalAveragePool";
        node_name = node_op_type + std::to_string(layer_count++);
      }
      else if (node.GetOpName() == "nn.avg_pool2d" || node.GetOpName() == "nn.max_pool2d") {
        node_op_type = node.GetOpName() == "nn.avg_pool2d" ? "AveragePool" : "MaxPool";
        node_name = node_op_type + std::to_string(layer_count++);
        auto pool_size = node.GetAttr<std::vector<std::string>>("pool_size");
        auto padding = node.GetAttr<std::vector<std::string>>("padding");
        auto strides = node.GetAttr<std::vector<std::string>>("strides");
        MWNNAttribute mwnn_attr_pool_size("pool_size", std::vector<int>({std::stoi(pool_size[0]), std::stoi(pool_size[1])}));
        node_attributes.emplace_back(mwnn_attr_pool_size);
        MWNNAttribute mwnn_attr_pad("pads", std::vector<int>({std::stoi(padding[0]), std::stoi(padding[1]), std::stoi(padding[2]), std::stoi(padding[3])}));
        node_attributes.emplace_back(mwnn_attr_pad);
        MWNNAttribute mwnn_attr_stride("strides", std::vector<int>({std::stoi(strides[0]), std::stoi(strides[1])}));
        node_attributes.emplace_back(mwnn_attr_stride);
      }
      else if (node.GetOpName() == "nn.lrn") {
        node_op_type = "LRN";
        node_name = node_op_type + std::to_string(layer_count++);
        auto alpha = node.GetAttr<std::vector<std::string>>("alpha");
        auto beta = node.GetAttr<std::vector<std::string>>("beta");
        auto size = node.GetAttr<std::vector<std::string>>("size");
        auto axis = node.GetAttr<std::vector<std::string>>("axis");
        auto bias = node.GetAttr<std::vector<std::string>>("bias");
        MWNNAttribute mwnn_attr_alpha("alpha", std::vector<int>({std::stoi(alpha[0])}));
        node_attributes.emplace_back(mwnn_attr_alpha);
        MWNNAttribute mwnn_attr_beta("beta", std::vector<int>({std::stoi(beta[0])}));
        node_attributes.emplace_back(mwnn_attr_beta);
        MWNNAttribute mwnn_attr_size("size", std::vector<int>({std::stoi(size[0])}));
        node_attributes.emplace_back(mwnn_attr_size);
        MWNNAttribute mwnn_attr_axis("axis", std::vector<int>({std::stoi(axis[0])}));
        node_attributes.emplace_back(mwnn_attr_axis);
        MWNNAttribute mwnn_attr_bias("bias", std::vector<int>({std::stoi(bias[0])}));
        node_attributes.emplace_back(mwnn_attr_bias);
      }
      else if (node.GetOpName() == "nn.batch_flatten") {
        node_op_type = "BatchFlatten";
        node_name = node_op_type + std::to_string(layer_count++);
      }
      else if (node.GetOpName() == "nn.dense") {
        node_op_type = "Dense";
        node_name = node_op_type + std::to_string(layer_count++);
        /*auto units = node.GetAttr<std::vector<std::string>>("units");
        MWNNAttribute mwnn_attr_units("units", std::vector<int>({std::stoi(units[0])}));
        node_attributes.emplace_back(mwnn_attr_units);*/
      }
      else if (node.GetOpName() == "nn.bias_add") {
        node_op_type = "BiasAdd";
        node_name = node_op_type + std::to_string(layer_count++);
      }
      else if (node.GetOpName() == "clip") {
        node_op_type = "Clip";
        node_name = node_op_type + std::to_string(layer_count++);
        auto min = node.GetAttr<std::vector<std::string>>("a_min");
        MWNNAttribute mwnn_attr_min("min", std::vector<int>({std::stoi(min[0])}));
        node_attributes.emplace_back(mwnn_attr_min);
        auto max = node.GetAttr<std::vector<std::string>>("a_max");
        MWNNAttribute mwnn_attr_max("max", std::vector<int>({std::stoi(max[0])}));
        node_attributes.emplace_back(mwnn_attr_max);
      }
      else if (node.GetOpName() == "squeeze") {
        node_op_type = "Squeeze";
        node_name = node_op_type + std::to_string(layer_count++);
        auto axis = node.GetAttr<std::vector<std::string>>("axis");
        MWNNAttribute mwnn_attr_axis("axis", std::vector<int>({std::stoi(axis[0])}));
        node_attributes.emplace_back(mwnn_attr_axis);
      }
      else if (node.GetOpName() == "transpose") {
        node_op_type = "Transpose";
        node_name = node_op_type + std::to_string(layer_count++);
      }
      else if (node.GetOpName() == "concatenate") {
        node_op_type = "Concat";
        node_name = node_op_type + std::to_string(layer_count++);
      }
      else if (node.GetOpName() == "max") {
        node_op_type = "Max";
        node_name = node_op_type + std::to_string(layer_count++);
      }
      else if (node.GetOpName() == "subtract") {
        node_op_type = "Subtract";
        node_name = node_op_type + std::to_string(layer_count++);
      }
      else if (node.GetOpName() == "exp") {
        node_op_type = "Exp";
        node_name = node_op_type + std::to_string(layer_count++);
      }
      else if (node.GetOpName() == "maximum") {
        node_op_type = "Maximum";
        node_name = node_op_type + std::to_string(layer_count++);
      }
      else if (node.GetOpName() == "minimum") {
        node_op_type = "Minimum";
        node_name = node_op_type + std::to_string(layer_count++);
      }
      else if (node.GetOpName() == "sum") {
        node_op_type = "Sum";
        node_name = node_op_type + std::to_string(layer_count++);
      }
      else if (node.GetOpName() == "divide") {
        node_op_type = "Divide";
        node_name = node_op_type + std::to_string(layer_count++);
      }
      else if (node.GetOpName() == "multiply") {
        node_op_type = "Mul";
        node_name = node_op_type + std::to_string(layer_count++);
      }
      else if (node.GetOpName() == "reshape") {
        node_op_type = "Reshape";
        node_name = node_op_type + std::to_string(layer_count++);
        std::vector<std::string> shape = node.GetAttr<std::vector<std::string>>("newshape");
        MWNNAttribute mwnn_attr_shape("shape", std::vector<int>({std::stoi(shape[0]), std::stoi(shape[1])}));
        node_attributes.emplace_back(mwnn_attr_shape);
      }
      else {
        std::cout << "\n Unsupported Op in MetaWareNN backend : " << node.GetOpName();
        exit(1);
      }
      /*std::cout << "\n ================================Node=============================\n";
      std::cout << "\n Name : " << node_name;
      std::cout << "\n Type : " << node_op_type;
      for (auto nip: node_inputs)
        std::cout << "\n Inputs : " << nip;
      for (auto nop: node_outputs)
        std::cout << "\n Outputs : " << nop;*/

      MWNNNode mwnn_node(node_name, node_op_type, node_attributes, node_inputs, node_outputs);
      mwnn_nodes.emplace_back(mwnn_node);
      auto node = mwnn_node.get_node();
      mwnn_graph_nodes[mwnn_node.get_name()] = std::move(node);
    }
  }
  std::vector<int> dims;
  auto data_type = static_cast<int>(dtypes[0].code);
  for(int m = 0; m < op_shape.size(); m++)
    for(int n = 0; n < op_shape[m].size(); n++) {
      dims.push_back(op_shape[m][n]);
    }
  //Add Outputs
  set_graph_outputs(op_name, dims, data_type);
}

void MWNNGraph::set_graph_initializers(std::string const_name, const DLTensor* data) {
  std::vector<int> dims(data->shape, data->shape + data->ndim);
  auto total_elements = std::accumulate(begin(dims), end(dims), 1, std::multiplies<int>());
  std::vector<float> tensor_vec(((float*)(data->data)), ((float*)(data->data)) + total_elements);

  MWNNTensor mwnn_tensor(const_name, dims, data->dtype.code, tensor_vec);
  mwnn_initializer_tensors.emplace_back(mwnn_tensor);
  mwnn_initializer_names.insert(mwnn_tensor.get_name());
  auto const_node = mwnn_tensor.get_constant_node();
  mwnn_graph_nodes[mwnn_tensor.get_name()] = std::move(const_node);
}

void MWNNGraph::set_graph_inputs(std::string name, const JSONGraphNode& node) {
  auto shapes = node.GetOpShape();
  auto dtypes = node.GetOpDataType();

  for (size_t i = 0; i < shapes.size(); ++i) {
    auto shape = shapes[i];
    int size = shape.size();
    std::vector<int> dims(size);
    for(int d = 0; d < size; d++)
      dims[d] = shape[d];
    auto data_type = static_cast<int>(dtypes[i].code);
    std::cout << "\nInput Name : " << name;
    std::cout << "\nInput Dims : ";
    for(int j=0; j<dims.size(); j++)
      std::cout << dims[j] << " ";
    std::cout << "\nInput Type : " << data_type;

    //Update the node name by assuming each graph input has unique JSONGraphNode
    //i loop runs only once in our case
    MWNNValueInfo mwnn_input(name, dims, data_type);
    std::string ip_name = mwnn_input.get_name();
    mwnn_graph_ip_names.emplace_back(ip_name);
    auto ip_node = mwnn_input.get_node();
    mwnn_graph_nodes[ip_name] = std::move(ip_node);
    //Fills Graph Input Tensor Details - Name, Dims
    MWNNTensor mwnn_ip_tensor(mwnn_input.get_name(), mwnn_input.get_type(), mwnn_input.get_dims());
    mwnn_graph_ip_tensors.emplace_back(mwnn_ip_tensor);
  }
}

void MWNNGraph::set_graph_outputs(std::string name, std::vector<int> dims, int type) {
    MWNNValueInfo mwnn_output(name, dims, type);
    mwnn_graph_op_names.emplace_back(mwnn_output.get_name());
    //Fills Graph Output Tensor Details - Name, Dims
    MWNNTensor mwnn_op_tensor(mwnn_output.get_name(), mwnn_output.get_type(), mwnn_output.get_dims());
    mwnn_graph_op_tensors.emplace_back(mwnn_op_tensor);

}
#endif
} //namespace metawarenn
