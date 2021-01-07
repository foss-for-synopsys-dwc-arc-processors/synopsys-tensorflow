#include "metawarenn_graph.h"

namespace metawarenn {
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
    mwnn_graph_nodes[mwnn_input.get_name()] = std::move(*ip_node);

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
        ::metawarenn::MWNNAttribute mwnn_attr_activation("activation", {conv_params->activation});
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

          //int out_height = ceil(in_height/conv_params->stride_height);
          //int out_width = ceil(in_width/conv_params->stride_width);
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
        /*std::cout << "\n ----------------------------------Attribute Conv params-----------------------------------";
        std::cout << "\n Conv stride height : " << conv_params->stride_height;
        std::cout << "\n Conv stride width : " << conv_params->stride_width;
        std::cout << "\n Conv padding : " << conv_params->padding;
        std::cout << "\n Conv activation : " << conv_params->activation;
        std::cout << "\n Conv dilation_width_factor : " << conv_params->dilation_width_factor;
        std::cout << "\n Conv dilation_height_factor : " << conv_params->dilation_height_factor;*/
      }
      else if (op_type == kTfLiteBuiltinDepthwiseConv2d) {
        node_op_type = "DepthwiseConv";
        node_name = node_op_type + std::to_string(node_index);
        const TfLiteDepthwiseConvParams* depthwise_conv_params = reinterpret_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);

        ::metawarenn::MWNNAttribute mwnn_attr_dilate("dilations", {depthwise_conv_params->dilation_height_factor, depthwise_conv_params->dilation_width_factor});
        node_attributes.emplace_back(mwnn_attr_dilate);
        ::metawarenn::MWNNAttribute mwnn_attr_stride("strides", {depthwise_conv_params->stride_height, depthwise_conv_params->stride_width});
        node_attributes.emplace_back(mwnn_attr_stride);
        ::metawarenn::MWNNAttribute mwnn_attr_activation("activation", {depthwise_conv_params->activation});
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

          //int out_height = ceil(in_height/depthwise_conv_params->stride_height);
          //int out_width = ceil(in_width/depthwise_conv_params->stride_width);
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
        /*std::cout << "\n ----------------------------------Attribute DepthWiseConv params-----------------------------------";
        std::cout << "\n DepthWiseConv stride height : " << depthwise_conv_params->stride_height;
        std::cout << "\n DepthWiseConv stride width : " << depthwise_conv_params->stride_width;
        std::cout << "\n DepthWiseConv padding : " << depthwise_conv_params->padding;
        std::cout << "\n DepthWiseConv activation : " << depthwise_conv_params->activation;
        std::cout << "\n DepthWiseConv dilation_width_factor : " << depthwise_conv_params->dilation_width_factor;
        std::cout << "\n DepthWiseConv dilation_height_factor : " << depthwise_conv_params->dilation_height_factor;
        std::cout << "\n DepthwiseConv depth_multiplier : " << depthwise_conv_params->depth_multiplier;*/
      }
      else if (op_type == kTfLiteBuiltinAveragePool2d) {
        node_op_type = "GlobalAveragePool";
        node_name = node_op_type + std::to_string(node_index);
        /*const TfLitePoolParams* pool_params = reinterpret_cast<const TfLitePoolParams*>(node->builtin_data);
        std::cout << "\n ----------------------------------Attribute Average_Pool params-----------------------------------";
        std::cout << "\n Global Average Pool padding : " << pool_params->padding;
        std::cout << "\n Global Average Pool stride_width : " << pool_params->stride_width;
        std::cout << "\n Global Average Pool stride_height : " << pool_params->stride_height;
        std::cout << "\n Global Average Pool filter_width : " << pool_params->filter_width;
        std::cout << "\n Global Average Pool filter_height : " << pool_params->filter_height;
        std::cout << "\n Global Average Pool activation : " << pool_params->activation;
        std::cout << "\n Global Average Pool computed->padding->width : " << pool_params->computed.padding.width;
        std::cout << "\n Global Average Pool computed->padding->height : " << pool_params->computed.padding.height;
        std::cout << "\n Global Average Pool computed->padding->width_offset : " << pool_params->computed.padding.width_offset;
        std::cout << "\n Global Average Pool computed->padding->height_offset : " << pool_params->computed.padding.height_offset;*/
      }
      else if (op_type == kTfLiteBuiltinAdd) {
        node_op_type = "Add";
        node_name = node_op_type + std::to_string(node_index);
        /*const TfLiteAddParams* add_params = reinterpret_cast<const TfLiteAddParams*>(node->builtin_data);
        std::cout << "\n ----------------------------------Attribute Add params-----------------------------------";
        std::cout << "\n Add activation : " << add_params->activation;*/
      }
      else if (op_type == kTfLiteBuiltinRelu) {
        node_op_type = "Relu";
        node_name = node_op_type + std::to_string(node_index);
      }
      else if (op_type == kTfLiteBuiltinReshape) {
        node_op_type = "Reshape";
        node_name = node_op_type + std::to_string(node_index);
        /*const TfLiteReshapeParams* reshape_params = reinterpret_cast<const TfLiteReshapeParams*>(node->builtin_data);
        std::cout << "\n ----------------------------------Attribute Reshape params-----------------------------------";
        std::cout << "\n Reshape num_dimensions: " << reshape_params->num_dimensions;
        for (int i = 0; i < 8; i++) {
          std::cout << "\n Reshape shape : " << reshape_params->shape[i];
        }*/
      }
      else if (op_type == kTfLiteBuiltinSoftmax) {
        node_op_type = "Softmax";
        node_name = node_op_type + std::to_string(node_index);
        /*const TfLiteSoftmaxParams* softmax_params = reinterpret_cast<const TfLiteSoftmaxParams*>(node->builtin_data);
        std::cout << "\n ----------------------------------Attribute Softmax params-----------------------------------";
        std::cout << "\n Softmax beta: " << softmax_params->beta;*/
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
            mwnn_graph_nodes[mwnn_tensor.get_name()] = std::move(*const_node);

            ::metawarenn::MWNNValueInfo mwnn_input(input_tensor.name, dims_vec, input_tensor.type);
            mwnn_inputs.emplace_back(mwnn_input);
            mwnn_initializer_names.insert(input_tensor.name);
        }
      }

      ::metawarenn::MWNNNode mwnn_node(node_name, node_op_type, node_attributes, node_inputs, node_outputs);
      mwnn_nodes.emplace_back(mwnn_node);
      auto op_node = mwnn_node.get_node();
      mwnn_graph_nodes[mwnn_node.get_name()] = std::move(*op_node);

    }
    std::cout << "\n----------------------------------------------------------------------------------------------------------------\n";
    /*for (auto& it : get_graph_initializers()) {
      std::cout << "\n Tensor Name: " << it.get_name();
    }
    for (auto& it : get_graph_nodes()) {
      std::cout << "\n Node Name: " << it.get_name();
    }*/
  }
} //namespace metawarenn
