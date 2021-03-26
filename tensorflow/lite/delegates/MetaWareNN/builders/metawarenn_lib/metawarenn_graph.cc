#include "metawarenn_graph.h"

namespace metawarenn {

//ONNXConstructor
MWNNGraph::MWNNGraph(GraphProto& onnx_graph_proto) {
  GraphProto graph_proto = onnx_graph_proto;
  name = graph_proto.name();

  for (auto tensor_proto : graph_proto.initializer()) {
    MWNNTensor mwnn_tensor(tensor_proto);
    mwnn_initializer_tensors.emplace_back(mwnn_tensor);
    mwnn_initializer_names.insert(mwnn_tensor.get_name());
    auto const_node = mwnn_tensor.get_constant_node();
    mwnn_graph_nodes[mwnn_tensor.get_name()] = std::move(const_node);
  }
  for (auto node_proto : graph_proto.node()) {
    MWNNNode mwnn_node(node_proto);
    mwnn_nodes.emplace_back(mwnn_node);
    auto node = mwnn_node.get_node();
    mwnn_graph_nodes[mwnn_node.get_name()] = std::move(node);
  }
  for (auto ip_value_info_proto : graph_proto.input()) {
    MWNNValueInfo mwnn_input(ip_value_info_proto);
    mwnn_inputs.emplace_back(mwnn_input);
    if(mwnn_initializer_names.count(mwnn_input.get_name()))
      continue;
    else {
      ip_name = mwnn_input.get_name();
      auto ip_node = mwnn_input.get_node();
      mwnn_graph_nodes[ip_name] = std::move(ip_node);
    }
  }
  for (auto op_value_info_proto : graph_proto.output()) {
    MWNNValueInfo mwnn_output(op_value_info_proto);
    mwnn_outputs.emplace_back(mwnn_output);
    op_name = mwnn_output.get_name();
  }
}

//TFConstructor
MWNNGraph::MWNNGraph(TfLiteContext* context, std::vector<int> subgraph_nodes_) {
  name = "MetaWareNN_NodeSubSet_1";
  std::cout << "\n----------------------------------------------------------------------------------------------------------------\n";
  std::cout << "\n MWNN Graph Name : " << get_name() << " with size as " << subgraph_nodes_.size() << " nodes";

  TfLiteNode* node;
  TfLiteRegistration* reg;

  //Set Graph Input Node
  context->GetNodeAndRegistration(context, 0, &node, &reg);
  int tensor_id = node->inputs->data[0];
  const auto& input_tensor = context->tensors[tensor_id];
  std::vector<int> dims_ip_vec(input_tensor.dims->data, input_tensor.dims->data + input_tensor.dims->size);
  ::metawarenn::MWNNValueInfo mwnn_input(input_tensor.name, dims_ip_vec, input_tensor.type);
  mwnn_inputs.emplace_back(mwnn_input);
  ip_name = input_tensor.name;

  auto ip_node = mwnn_input.get_node();
  mwnn_graph_nodes[mwnn_input.get_name()] = std::move(ip_node);

  //Set Graph Output Node
  context->GetNodeAndRegistration(context, (subgraph_nodes_.size() - 1), &node, &reg);
  tensor_id = node->outputs->data[0];
  const auto& output_tensor = context->tensors[tensor_id];
  std::vector<int> dims_op_vec(output_tensor.dims->data, output_tensor.dims->data + output_tensor.dims->size);
  ::metawarenn::MWNNValueInfo mwnn_output(output_tensor.name, dims_op_vec, output_tensor.type);
  mwnn_outputs.emplace_back(mwnn_output);
  op_name = output_tensor.name;

  for (size_t node_index = 0; node_index < subgraph_nodes_.size(); node_index++) {
    std::cout << "\n -------------------------------------------------------------------------------------------------------------";
    TfLiteNode* node;
    TfLiteRegistration* reg;
    const auto status = context->GetNodeAndRegistration(context, node_index, &node, &reg);
    auto op_type = reg->builtin_code;

    std::string node_name;
    std::string node_op_type;
    std::vector<std::string> node_inputs;
    std::vector<std::string> node_outputs;
    std::vector<::metawarenn::MWNNAttribute> node_attributes;

    //Op Names are added to follow the same pattern like in ONNX as of now.
    if (op_type == kTfLiteBuiltinConv2d) {
      node_op_type = "Conv";
      node_name = node_op_type + std::to_string(node_index);
      const TfLiteConvParams* conv_params = reinterpret_cast<const TfLiteConvParams*>(node->builtin_data);

      ::metawarenn::MWNNAttribute mwnn_attr_dilate("dilations", {conv_params->dilation_height_factor, conv_params->dilation_width_factor});
      node_attributes.emplace_back(mwnn_attr_dilate);
      ::metawarenn::MWNNAttribute mwnn_attr_stride("strides", {conv_params->stride_height, conv_params->stride_width});
      node_attributes.emplace_back(mwnn_attr_stride);

      int activation_type;
      if(conv_params->activation == ::tflite::ActivationFunctionType_NONE)
        activation_type = ActivationType::Activation_None;
      else if(conv_params->activation == ::tflite::ActivationFunctionType_RELU)
        activation_type = ActivationType::Activation_Relu;
      else if(conv_params->activation == ::tflite::ActivationFunctionType_RELU6)
        activation_type = ActivationType::Activation_Relu6;

      ::metawarenn::MWNNAttribute mwnn_attr_activation("activation", {activation_type});
      node_attributes.emplace_back(mwnn_attr_activation);

      if(conv_params->padding == kTfLitePaddingSame) {
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

        ::metawarenn::MWNNAttribute mwnn_attr_pad("pads", {pad_top, pad_left, pad_bottom, pad_right});
        node_attributes.emplace_back(mwnn_attr_pad);
      }
      else {
        ::metawarenn::MWNNAttribute mwnn_attr_pad("pads", {0, 0, 0, 0});
        node_attributes.emplace_back(mwnn_attr_pad);
      }
    }
    else if (op_type == kTfLiteBuiltinDepthwiseConv2d) {
      node_op_type = "DepthwiseConv";
      node_name = node_op_type + std::to_string(node_index);
      const TfLiteDepthwiseConvParams* depthwise_conv_params = reinterpret_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);

      ::metawarenn::MWNNAttribute mwnn_attr_dilate("dilations", {depthwise_conv_params->dilation_height_factor, depthwise_conv_params->dilation_width_factor});
      node_attributes.emplace_back(mwnn_attr_dilate);
      ::metawarenn::MWNNAttribute mwnn_attr_stride("strides", {depthwise_conv_params->stride_height, depthwise_conv_params->stride_width});
      node_attributes.emplace_back(mwnn_attr_stride);

      int activation_type;
      if(depthwise_conv_params->activation == ::tflite::ActivationFunctionType_NONE)
        activation_type = ActivationType::Activation_None;
      else if(depthwise_conv_params->activation == ::tflite::ActivationFunctionType_RELU)
        activation_type = ActivationType::Activation_Relu;
      else if(depthwise_conv_params->activation == ::tflite::ActivationFunctionType_RELU6)
        activation_type = ActivationType::Activation_Relu6;

      ::metawarenn::MWNNAttribute mwnn_attr_activation("activation", {activation_type});
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

        ::metawarenn::MWNNAttribute mwnn_attr_pad("pads", {pad_top, pad_left, pad_bottom, pad_right});
        node_attributes.emplace_back(mwnn_attr_pad);
      }
      else {
        ::metawarenn::MWNNAttribute mwnn_attr_pad("pads", {0, 0, 0, 0});
        node_attributes.emplace_back(mwnn_attr_pad);
      }
    }
    else if (op_type == kTfLiteBuiltinAveragePool2d) {
      node_op_type = "GlobalAveragePool";
      node_name = node_op_type + std::to_string(node_index);
    }
    else if (op_type == kTfLiteBuiltinAdd) {
      node_op_type = "Add";
      node_name = node_op_type + std::to_string(node_index);
    }
    else if (op_type == kTfLiteBuiltinRelu) {
      node_op_type = "Relu";
      node_name = node_op_type + std::to_string(node_index);
    }
    else if (op_type == kTfLiteBuiltinReshape) {
      node_op_type = "Reshape";
      node_name = node_op_type + std::to_string(node_index);
    }
    else if (op_type == kTfLiteBuiltinSoftmax) {
      node_op_type = "Softmax";
      node_name = node_op_type + std::to_string(node_index);
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
          mwnn_inputs.emplace_back(mwnn_input);
          mwnn_initializer_names.insert(input_tensor.name);
      }
    }
    ::metawarenn::MWNNNode mwnn_node(node_name, node_op_type, node_attributes, node_inputs, node_outputs);
    mwnn_nodes.emplace_back(mwnn_node);
    auto op_node = mwnn_node.get_node();
    mwnn_graph_nodes[mwnn_node.get_name()] = std::move(op_node);
  }
}
#if GLOW
//GLOWConstructor
MWNNGraph::MWNNGraph(Function *F) {
  name = std::string(F->getName());
  LOG(INFO) << "Function name: " << name;
  GraphPostOrderVisitor visitor(*F);
  auto node_list = visitor.getPostOrder();
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
            auto dilations = conv_node->getDilation();
            auto strides = conv_node->getStrides();
            auto pads = conv_node->getPads();
            auto group = conv_node->getGroup();
            metawarenn::MWNNAttribute mwnn_attr_dilate("dilations", {int(dilations[0]), int(dilations[1])});
            node_attributes.emplace_back(mwnn_attr_dilate);
            metawarenn::MWNNAttribute mwnn_attr_stride("strides", {int(strides[0]), int(strides[0])});
            node_attributes.emplace_back(mwnn_attr_stride);
            metawarenn::MWNNAttribute mwnn_attr_pad("pads", {int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])});
            node_attributes.emplace_back(mwnn_attr_pad);
            metawarenn::MWNNAttribute mwnn_attr_group("group", {int(group)});
            node_attributes.emplace_back(mwnn_attr_group);
            metawarenn::MWNNAttribute mwnn_attribute("activation", {0});
            node_attributes.emplace_back(mwnn_attribute);
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
            std::vector<int> dims_(dims.size());
            int i = 0;
            for(auto dim: dims){
              dims_vec[i] = dim;
              dims_[i++] = dim;
            }
            metawarenn::MWNNTensor mwnn_reshape_tensor(initializer_name, dims_, ElemKind::Int32ITy, dims_vec);
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
        default:
            break;
        }
        metawarenn::MWNNNode mwnn_node(node_name, node_op_type, node_attributes, node_inputs, node_outputs);
        mwnn_nodes.emplace_back(mwnn_node);
        auto op_node = mwnn_node.get_node();
        mwnn_graph_nodes[mwnn_node.get_name()] = std::move(op_node);
      }
  }
  // Graph input and output handling
  auto &nodes = F->getNodes();
  auto &first_node = nodes.front();
  auto &last_node = nodes.back();
  auto input_name = std::string(first_node.getNthInput(0).getNode()->getName());
  auto output_name = std::string(last_node.getNthResult(0).getNode()->getName());
  ip_name = input_name;
  op_name = output_name;
  for (auto *v : F->getParent()->getPlaceholders()) {
    auto glow_dims = v->getType()->dims();
    auto data_type = v->getType()->getElementType();
    int size = glow_dims.size();
    std::vector<int> dims(size);
    for(int i = 0; i < size; i++)
    {
      dims[i] = int(glow_dims[i]);
    }
    if(v->getName().equals(input_name))
    {
      metawarenn::MWNNValueInfo mwnn_input(input_name, dims, data_type);
      mwnn_inputs.emplace_back(mwnn_input);
      mwnn_initializer_names.insert(input_name);
    }
    else
    {
      metawarenn::MWNNValueInfo mwnn_output(input_name, dims, data_type);
      mwnn_outputs.emplace_back(mwnn_output);
      mwnn_initializer_names.insert(output_name);
    }
  }
}
#endif

} //namespace metawarenn
