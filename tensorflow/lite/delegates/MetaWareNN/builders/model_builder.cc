#include "model_builder.h"
#include <cstdlib>
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
// helper function `GetTensorData` is defined in `tensor_ctypes.h`

namespace tflite {
namespace delegates {
namespace metawarenn {

void ModelBuilder::CreateMWNNNode(std::shared_ptr<::metawarenn::Graph> graph_ptr_,
                                  const std::string &node_name_,
                                  const std::string &node_op_type_,
                                  const std::vector<::metawarenn::Attribute> &node_attributes_,
                                  const std::vector<std::string> &node_inputs_,
                                  const std::vector<std::string> &node_outputs_) {
  ::metawarenn::Node m_node(node_name_, node_op_type_, node_attributes_, node_inputs_, node_outputs_);
  graph_ptr_->set_graph_nodes(m_node);

  std::cout << "\n ================================Node=============================\n";
  std::cout << "\n Name : " << node_name_;
  std::cout << "\n Type : " << node_op_type_;
  for (auto nip: node_inputs_)
    std::cout << "\n Inputs : " << nip;
  for (auto nop: node_outputs_)
    std::cout << "\n Outputs : " << nop;
}

void ModelBuilder::CreateMWNNQuantParams(std::shared_ptr<::metawarenn::Graph> graph_ptr_,
                                         TfLiteTensor tensor) {
  std::string scale_name = tensor.name + std::string("_scale");
  std::vector<float> tensor_vec_scale = {tensor.params.scale};
  ::metawarenn::Tensor scale_tensor(scale_name, std::vector<int>({tensor_vec_scale.size()}), ::metawarenn::Element::ElementType::kFloat, tensor_vec_scale);
  graph_ptr_->set_graph_initializers(scale_tensor);
  graph_ptr_->initializer_names_.insert(scale_name);

  std::string zp_name = tensor.name + std::string("_zero_point");
  std::vector<int32_t> tensor_vec_zp = {tensor.params.zero_point};
  std::cout << "\n get_mwnn_type_tf(tensor.type): " << (int)get_mwnn_type_tf(tensor.type);
  ::metawarenn::Tensor zp_tensor(zp_name, std::vector<int>({tensor_vec_zp.size()}), get_mwnn_type_tf(tensor.type), tensor_vec_zp);
  graph_ptr_->set_graph_initializers(zp_tensor);
  graph_ptr_->initializer_names_.insert(zp_name);
}

template<class T>
void ModelBuilder::parse_input(TfLiteTensor input_tensor, std::shared_ptr<::metawarenn::Graph> graph_ptr) {
  if (input_tensor.allocation_type == kTfLiteMmapRo) {
    std::vector<int> dims_vec(input_tensor.dims->data, input_tensor.dims->data + input_tensor.dims->size);
    auto num_tensor_elements = std::accumulate(begin(dims_vec), end(dims_vec), 1, std::multiplies<int>());
    std::vector<T> tensor_vec;
    if(input_tensor.type == kTfLiteFloat32) {
      auto *ip_data = GetTensorData<float>(&input_tensor);
      tensor_vec = {ip_data, ip_data+num_tensor_elements};
    }
    else if(input_tensor.type == kTfLiteUInt8) {
      auto *ip_data = GetTensorData<uint8_t>(&input_tensor);
      tensor_vec = {ip_data, ip_data+num_tensor_elements};
    }
    else if(input_tensor.type == kTfLiteInt32) {
      auto *ip_data = GetTensorData<int32_t>(&input_tensor);
      tensor_vec = {ip_data, ip_data+num_tensor_elements};
    }
    ::metawarenn::Tensor m_tensor(input_tensor.name, dims_vec, get_mwnn_type_tf(input_tensor.type), tensor_vec);
    graph_ptr->set_graph_initializers(m_tensor);
    graph_ptr->initializer_names_.insert(input_tensor.name);
  }
}

ModelBuilder::ModelBuilder(std::vector<int> nodes)
    : subgraph_nodes_(nodes) {}

std::shared_ptr<::metawarenn::Graph> ModelBuilder::BuildGraph(TfLiteContext* context, std::string subgraph_name) {
  std::cout<<"\nBuildGraph!!"<<std::endl;

  /*Create MetaWareNN High Level Graph Representation from TFLite SubGraph Nodes*/
  std::shared_ptr<::metawarenn::Graph> graph_ptr = std::make_shared<::metawarenn::Graph>();
  graph_ptr->set_name(subgraph_name);

  std::cout << "\n----------------------------------------------------------------------------------------------------------------\n";
  std::cout << "\n MWNN Graph Name : " << graph_ptr->get_name() << " with size as " << subgraph_nodes_.size() << " nodes";

  TfLiteNode* node;
  TfLiteRegistration* reg;
  std::string node_name;
  std::string node_op_type;
  std::string activation_node_name;
  std::string activation_node_op_type;
  std::vector<std::string> node_inputs;
  std::vector<std::string> node_outputs;
  std::vector<::metawarenn::Attribute> node_attributes;
  std::map<std::string, std::string> quant_ip_mapper;
  std::map<std::string, std::string> quant_op_mapper;

  //Set Graph Input Node
  context->GetNodeAndRegistration(context, subgraph_nodes_[0], &node, &reg);
  int tensor_id = node->inputs->data[0];
  TfLiteTensor input_tensor = context->tensors[tensor_id];
  std::vector<int> dims_ip_vec(input_tensor.dims->data, input_tensor.dims->data + input_tensor.dims->size);
  graph_ptr->set_graph_ip_names(input_tensor.name);
  //Fills Graph Input Tensor Details - Name, Dims
  ::metawarenn::Tensor m_ip_tensor(input_tensor.name, get_mwnn_type_tf(input_tensor.type), dims_ip_vec);
  graph_ptr->set_graph_ip_tensor(m_ip_tensor);

  if(input_tensor.type == kTfLiteUInt8) {
    CreateMWNNQuantParams(graph_ptr, input_tensor);
    node_inputs.clear(); node_outputs.clear();
    node_op_type = "DequantizeLinear";
    node_name = node_op_type + "_" + input_tensor.name;
    node_inputs.push_back(input_tensor.name);
    node_inputs.push_back(input_tensor.name + std::string("_scale"));
    node_inputs.push_back(input_tensor.name + std::string("_zero_point"));
    node_outputs.push_back(node_name);
    CreateMWNNNode(graph_ptr, node_name, node_op_type, node_attributes, node_inputs, node_outputs);
    quant_ip_mapper[input_tensor.name] = node_outputs[0];
  }

  //Set Graph Output Node
  context->GetNodeAndRegistration(context, subgraph_nodes_[subgraph_nodes_.size()-1], &node, &reg);
  tensor_id = node->outputs->data[0];
  TfLiteTensor output_tensor = context->tensors[tensor_id];
  std::vector<int> dims_op_vec(output_tensor.dims->data, output_tensor.dims->data + output_tensor.dims->size);

  if(output_tensor.type == kTfLiteUInt8) {
    CreateMWNNQuantParams(graph_ptr, output_tensor);
    node_inputs.clear(); node_outputs.clear();
    node_op_type = "QuantizeLinear";
    node_name = node_op_type + "_" + input_tensor.name;
    node_inputs.push_back(output_tensor.name);
    node_inputs.push_back(output_tensor.name + std::string("_scale"));
    node_inputs.push_back(output_tensor.name + std::string("_zero_point"));
    node_outputs.push_back(node_name);
    CreateMWNNNode(graph_ptr, node_name, node_op_type, node_attributes, node_inputs, node_outputs);
    quant_op_mapper[output_tensor.name] = node_outputs[0];

    graph_ptr->set_graph_op_names(node_outputs[0]);
    //Fills Graph Output Tensor Details - Name, Dims
    ::metawarenn::Tensor m_op_tensor(node_outputs[0], ::metawarenn::Element::ElementType::kUint8, dims_op_vec);
     graph_ptr->set_graph_op_tensor(m_op_tensor);
  }
  else {
    graph_ptr->set_graph_op_names(output_tensor.name);
    //Fills Graph Output Tensor Details - Name, Dims
    ::metawarenn::Tensor m_op_tensor(output_tensor.name, get_mwnn_type_tf(output_tensor.type), dims_op_vec);
    graph_ptr->set_graph_op_tensor(m_op_tensor);
  }


  // If multiple layers share the constant tensor, we only keep one copy in MWNN
  //auto const_names = map<string, int>();
  std::map<std::string, int> const_names;

  for (size_t node_index = 0; node_index < subgraph_nodes_.size(); node_index++) {
    std::cout << "\n -------------------------------------------------------------------------------------------------------------";
    TfLiteNode* node;
    TfLiteRegistration* reg;
    const auto status = context->GetNodeAndRegistration(context, subgraph_nodes_[node_index], &node, &reg);
    auto op_type = reg->builtin_code;
    node_name = "";
    node_op_type = "";
    activation_node_name = "";
    activation_node_op_type = "";
    node_inputs.clear();
    node_outputs.clear();
    node_attributes.clear();

    for (int i = 0; i < node->inputs->size; ++i) {
      const int tensor_id = node->inputs->data[i];
      TfLiteTensor input_tensor = context->tensors[tensor_id];
      if (input_tensor.allocation_type == kTfLiteMmapRo) {
        if(input_tensor.type == kTfLiteFloat32) {
          parse_input<float>(input_tensor, graph_ptr);
        }
        else if(input_tensor.type == kTfLiteUInt8 || input_tensor.type == kTfLiteInt32) {
          parse_input<int32_t>(input_tensor, graph_ptr);
          auto q_params = input_tensor.params;
          if(q_params.scale || q_params.zero_point) {
            CreateMWNNQuantParams(graph_ptr, input_tensor);
            node_inputs.clear(); node_outputs.clear();
            node_op_type = "DequantizeLinear";
            node_name = node_op_type + "_" + input_tensor.name;
            node_inputs.push_back(input_tensor.name);
            node_inputs.push_back(input_tensor.name + std::string("_scale"));
            node_inputs.push_back(input_tensor.name + std::string("_zero_point"));
            node_outputs.push_back(node_name);
            CreateMWNNNode(graph_ptr, node_name, node_op_type, node_attributes, node_inputs, node_outputs);
            quant_ip_mapper[input_tensor.name] = node_outputs[0];
          }
        }
        else {
          std::cout << "\n Unhandled Type " << (int)get_mwnn_type_tf(input_tensor.type);
          exit(1);
        }
      }
    }
    node_inputs.clear(); node_outputs.clear();
    for (int i = 0; i < node->outputs->size; ++i) {
      const int tensor_id = node->outputs->data[i];
      TfLiteTensor output_tensor = context->tensors[tensor_id];
      auto itr = quant_op_mapper.find(output_tensor.name);
      //Don't QDQ nodes for last op node - handled previously with Q node
      if(output_tensor.type == kTfLiteUInt8 && itr == quant_op_mapper.end()) {
        auto q_params = output_tensor.params;
        if(q_params.scale || q_params.zero_point) {
          CreateMWNNQuantParams(graph_ptr, output_tensor);
          node_inputs.clear(); node_outputs.clear();
          node_op_type = "QuantizeLinear";
          node_name = node_op_type + "_" + output_tensor.name;
          node_inputs.push_back(output_tensor.name);
          node_inputs.push_back(output_tensor.name + std::string("_scale"));
          node_inputs.push_back(output_tensor.name + std::string("_zero_point"));
          node_outputs.push_back(node_name);
          CreateMWNNNode(graph_ptr, node_name, node_op_type, node_attributes, node_inputs, node_outputs);

          node_op_type = "DequantizeLinear";
          node_name = node_op_type + "_" + output_tensor.name;
          node_inputs.clear();
          node_inputs.push_back(node_outputs[0]);
          node_inputs.push_back(output_tensor.name + std::string("_scale"));
          node_inputs.push_back(output_tensor.name + std::string("_zero_point"));
          node_outputs.clear();
          node_outputs.push_back(node_name);
          CreateMWNNNode(graph_ptr, node_name, node_op_type, node_attributes, node_inputs, node_outputs);
          quant_ip_mapper[output_tensor.name] = node_outputs[0];
        }
      }
    }
    node_inputs.clear(); node_outputs.clear();
    for (int i = 0; i < node->inputs->size; ++i) {
      const int tensor_id = node->inputs->data[i];
       auto itr = quant_ip_mapper.find(context->tensors[tensor_id].name);
       if(itr != quant_ip_mapper.end())
         node_inputs.emplace_back(itr->second);
       else
         node_inputs.emplace_back(context->tensors[tensor_id].name);
    }
    for (int i = 0; i < node->outputs->size; ++i) {
      const int tensor_id = node->outputs->data[i];
      node_outputs.emplace_back(context->tensors[tensor_id].name);
    }
    //Op Names are added to follow the same pattern like in ONNX as of now.
    if (op_type == kTfLiteBuiltinConv2d) {
      node_op_type = "Conv";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteConvParams* conv_params = reinterpret_cast<const TfLiteConvParams*>(node->builtin_data);
      const int weight_tensor_id = node->inputs->data[1];
      const auto& weight_tensor = context->tensors[weight_tensor_id];

      ::metawarenn::Attribute attr_dilate("dilations", std::vector<int64_t>{conv_params->dilation_height_factor, conv_params->dilation_width_factor});
      node_attributes.emplace_back(attr_dilate);
      ::metawarenn::Attribute attr_kernel_shape("kernel_shape", std::vector<int64_t>{weight_tensor.dims->data[1], weight_tensor.dims->data[2]});
      node_attributes.emplace_back(attr_kernel_shape);
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
        ::metawarenn::Attribute attr_pad("pads", std::vector<int64_t>{pad_top, pad_left, pad_bottom, pad_right});
        node_attributes.emplace_back(attr_pad);
      }
      else {
        ::metawarenn::Attribute attr_pad("pads", std::vector<int64_t>{0, 0, 0, 0});
        node_attributes.emplace_back(attr_pad);
      }
      ::metawarenn::Attribute attr_stride("strides", std::vector<int64_t>{conv_params->stride_height, conv_params->stride_width});
      node_attributes.emplace_back(attr_stride);
      if(conv_params->activation == ::tflite::ActivationFunctionType_RELU) {
        activation_node_op_type = "Relu";
        activation_node_name = activation_node_op_type + std::to_string(subgraph_nodes_[node_index]);
      }
      else if(conv_params->activation == ::tflite::ActivationFunctionType_RELU6) {
        activation_node_op_type = "Clip";
        activation_node_name = activation_node_op_type + std::to_string(subgraph_nodes_[node_index]);
      }
    }
    else if (op_type == kTfLiteBuiltinDepthwiseConv2d) {
      node_op_type = "Conv";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteDepthwiseConvParams* depthwise_conv_params = reinterpret_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);
      const int ip_tensor_id = node->inputs->data[0];
      int64_t group =  context->tensors[ip_tensor_id].dims->data[3];
      const int weight_tensor_id = node->inputs->data[1];
      const auto& weight_tensor = context->tensors[weight_tensor_id];
      int depth_multiplier = depthwise_conv_params->depth_multiplier;

      ::metawarenn::Attribute attr_dilate("dilations", std::vector<int64_t>{depthwise_conv_params->dilation_height_factor, depthwise_conv_params->dilation_width_factor});
      node_attributes.emplace_back(attr_dilate);
      ::metawarenn::Attribute attr_group("group", group);
      node_attributes.emplace_back(attr_group);
      ::metawarenn::Attribute attr_kernel_shape("kernel_shape", std::vector<int64_t>{weight_tensor.dims->data[1], weight_tensor.dims->data[2]});
      node_attributes.emplace_back(attr_kernel_shape);
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
        ::metawarenn::Attribute attr_pad("pads",std::vector<int64_t> {pad_top, pad_left, pad_bottom, pad_right});
        node_attributes.emplace_back(attr_pad);
      }
      else {
        ::metawarenn::Attribute attr_pad("pads", std::vector<int64_t>{0, 0, 0, 0});
        node_attributes.emplace_back(attr_pad);
      }
      ::metawarenn::Attribute attr_stride("strides", std::vector<int64_t>{depthwise_conv_params->stride_height, depthwise_conv_params->stride_width});
      node_attributes.emplace_back(attr_stride);
      if(depthwise_conv_params->activation == ::tflite::ActivationFunctionType_RELU) {
        activation_node_op_type = "Relu";
        activation_node_name = activation_node_op_type + std::to_string(subgraph_nodes_[node_index]);
      }
      else if(depthwise_conv_params->activation == ::tflite::ActivationFunctionType_RELU6) {
        activation_node_op_type = "Clip";
        activation_node_name = activation_node_op_type + std::to_string(subgraph_nodes_[node_index]);
      }
    }
    else if (op_type == kTfLiteBuiltinAdd) {
      node_op_type = "Add";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteAddParams* add_params = reinterpret_cast<const TfLiteAddParams*>(node->builtin_data);
      if(add_params->activation == ::tflite::ActivationFunctionType_RELU) {
        activation_node_op_type = "Relu";
        activation_node_name = activation_node_op_type + std::to_string(subgraph_nodes_[node_index]);
      }
      else if(add_params->activation == ::tflite::ActivationFunctionType_RELU6) {
        activation_node_op_type = "Clip";
        activation_node_name = activation_node_op_type + std::to_string(subgraph_nodes_[node_index]);
      }
    }
    else if (op_type == kTfLiteBuiltinRelu) {
      node_op_type = "Relu";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
    }
    else if (op_type == kTfLiteBuiltinMaxPool2d || op_type == kTfLiteBuiltinAveragePool2d) {
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
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      if(node_op_type != "GlobalAveragePool") {
        ::metawarenn::Attribute attr_kernel_shape("kernel_shape", std::vector<int64_t>{pool_params->filter_height, pool_params->filter_width});
        node_attributes.emplace_back(attr_kernel_shape);
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

          ::metawarenn::Attribute attr_pad("pads", std::vector<int64_t>{pad_top, pad_left, pad_bottom, pad_right});
          node_attributes.emplace_back(attr_pad);
        }
        else {
          ::metawarenn::Attribute attr_pad("pads", std::vector<int64_t>{0, 0, 0, 0});
          node_attributes.emplace_back(attr_pad);
        }
        ::metawarenn::Attribute attr_stride("strides", std::vector<int64_t>{pool_params->stride_height, pool_params->stride_width});
        node_attributes.emplace_back(attr_stride);
      }
    }
    else if (op_type == kTfLiteBuiltinMul) {
      node_op_type = "Mul";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
    }
    else if (op_type == kTfLiteBuiltinConcatenation) {
      node_op_type = "Concat";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteConcatenationParams* concat_params = reinterpret_cast<const TfLiteConcatenationParams*>(node->builtin_data);
      if(concat_params->activation != kTfLiteActNone) {
        std::cout << "\n Unsupported Fused Activation Present in Concat Layer "<< node_name << " Add Activation node to Handle!!!";
        exit(1);
      }
      ::metawarenn::Attribute attr_axis("axis", (int64_t)(concat_params->axis-2));//HWC to CHW - Concat along Channel Dimension
      node_attributes.emplace_back(attr_axis);
    }
    else if (op_type == kTfLiteBuiltinMean) {
      const TfLiteReducerParams* reduce_params = reinterpret_cast<const TfLiteReducerParams*>(node->builtin_data);
      bool keepdims = reduce_params->keep_dims;

      const int axis_tensor_id = node->inputs->data[1]; // The axis to be reduced.
      const auto& axis_tensor = context->tensors[axis_tensor_id];
      std::vector<int> dims_vec(axis_tensor.dims->data, axis_tensor.dims->data + axis_tensor.dims->size);
      auto num_tensor_elements = std::accumulate(begin(dims_vec), end(dims_vec), 1, std::multiplies<int>());
      std::vector<int> tensor_vec(axis_tensor.data.i32, axis_tensor.data.i32 + num_tensor_elements);

      if(tensor_vec[0] == 1 && tensor_vec[1] == 2) {
        node_op_type = "GlobalAveragePool";
        node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
        node_inputs.pop_back();
      }
      else {
        node_op_type = "Mean";
        node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
        /*std::vector<int> int_axis(axis.size());
        std::transform(axis.begin(), axis.end(), std::back_inserter(int_axis),
                      [](const std::string& str) { return std::stoi(str); });
        metawarenn::Attribute attr_axis("axis", tensor_vec);
        node_attributes.emplace_back(attr_axis);*/
      }
    }
    else if (op_type == kTfLiteBuiltinFullyConnected) {
      node_op_type = "Gemm";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      //parse attributes
      const TfLiteFullyConnectedParams* fc_params = reinterpret_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);
      assert(fc_params->asymmetric_quantize_inputs == false); // bool
      assert(fc_params->keep_num_dims == false); // bool
      assert(fc_params->weights_format == 0); // TfLiteFullyConnectedWeightsFormat
      assert(fc_params->activation == kTfLiteActNone);

      const int input_tensor_id0 = node->inputs->data[0];
      const auto& input_tensor0 = context->tensors[input_tensor_id0];
      std::vector<int> ip_dims0(input_tensor0.dims->data, input_tensor0.dims->data + input_tensor0.dims->size);
      const int input_tensor_id1 = node->inputs->data[1];
      const auto& input_tensor1 = context->tensors[input_tensor_id1];
      std::vector<int> ip_dims1(input_tensor1.dims->data, input_tensor1.dims->data + input_tensor1.dims->size);

      if(ip_dims0[ip_dims0.size()-1] != ip_dims1[0]) {
        ::metawarenn::Attribute attr_transB("transB", (int64_t)1);
        node_attributes.emplace_back(attr_transB);
      }

      //GEMM Requires 2D Input - Insert Reshape Node
      std::string reshape_node_op_type = "Reshape";
      std::string reshape_node_name = reshape_node_op_type + std::to_string(subgraph_nodes_[node_index]);
      std::vector<::metawarenn::Attribute> reshape_node_attributes;
      std::vector<std::string> reshape_node_inputs;
      std::vector<std::string> reshape_node_outputs;

      std::string reshape_ip_name = reshape_node_name + "_ip";
      std::vector<int64_t> tensor_vec = {-1, ip_dims0[ip_dims0.size()-1]};

      ::metawarenn::Tensor reshape_tensor(reshape_ip_name, std::vector<int>({tensor_vec.size()}), ::metawarenn::Element::ElementType::kInt64, tensor_vec);
      graph_ptr->set_graph_initializers(reshape_tensor);
      graph_ptr->initializer_names_.insert(reshape_ip_name);

      reshape_node_inputs.emplace_back(node_inputs[0]);
      reshape_node_inputs.emplace_back(reshape_ip_name);
      reshape_node_outputs.emplace_back(reshape_node_name + "_output");

      //Creates New Reshape Node to Update the GEMM Input Tensor
      CreateMWNNNode(graph_ptr, reshape_node_name, reshape_node_op_type, reshape_node_attributes, reshape_node_inputs, reshape_node_outputs);

      //Updates the GEMM Input with Reshape Node Output
      node_inputs[0] = reshape_node_name + "_output";
    }
    else if (op_type == kTfLiteBuiltinSplit) {
      node_op_type = "Split";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteSplitParams* split_params = reinterpret_cast<const TfLiteSplitParams*>(node->builtin_data);

      //Parse the Input Axis and convert to Attribute
      const int input_tensor_id0 = node->inputs->data[0];
      const auto& input_tensor0 = context->tensors[input_tensor_id0];
      int64_t axis = input_tensor0.data.i32[0] - 2; //HWC to CHW conversion
      ::metawarenn::Attribute attr_axis("axis", axis);
      node_attributes.emplace_back(attr_axis);
      //Get the Input Tensor details
      const int input_tensor_id1 = node->inputs->data[1];
      const auto& input_tensor1 = context->tensors[input_tensor_id1];
      //Removes the existing inputs and updates the tensor input in 0th position
      node_inputs.pop_back();
      node_inputs.pop_back();
      node_inputs.emplace_back(input_tensor1.name);
    }
    else if (op_type == kTfLiteBuiltinPad) {
      node_op_type = "Pad";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);

      //Parse the Input 1 value and do the CHW conversion along with int type handling
      std::string pad_ip_name = node_name + "_ip";
      const int pad_tensor_id = node->inputs->data[1];
      const auto& pad_tensor = context->tensors[pad_tensor_id];
      std::vector<int> dims_vec(pad_tensor.dims->data, pad_tensor.dims->data + pad_tensor.dims->size);
      auto num_tensor_elements = std::accumulate(begin(dims_vec), end(dims_vec), 1, std::multiplies<int>());
      std::vector<int> tensor_vec(pad_tensor.data.i32, pad_tensor.data.i32 + num_tensor_elements);

      std::vector<int> dims{num_tensor_elements};
      std::vector<int64_t> new_tensor_vec;

      std::vector<int> NDimValue{tensor_vec[0], tensor_vec[1]};
      std::vector<int> HDimValue{tensor_vec[2], tensor_vec[3]};
      std::vector<int> WDimValue{tensor_vec[4], tensor_vec[5]};
      std::vector<int> CDimValue{tensor_vec[7], tensor_vec[7]};
      //Pad layer NHWC --> NCHW
      //TFLite Format (NStart, NEnd, HStart, HEnd, WStart, WEnd, CStart, CEnd) (0, 1, 2, 3, 4, 5, 6, 7)
      //ONNX Format   (NStart, CStart, HStart, WStart, NEnd, CEnd, HEnd, WEnd) (0, 6, 2, 4, 1, 7, 3, 5)
      new_tensor_vec.emplace_back(NDimValue[0]);
      new_tensor_vec.emplace_back(CDimValue[0]);
      new_tensor_vec.emplace_back(HDimValue[0]);
      new_tensor_vec.emplace_back(WDimValue[0]);
      new_tensor_vec.emplace_back(NDimValue[1]);
      new_tensor_vec.emplace_back(CDimValue[1]);
      new_tensor_vec.emplace_back(HDimValue[1]);
      new_tensor_vec.emplace_back(WDimValue[1]);

      ::metawarenn::Tensor pad_ip_tensor(pad_ip_name, dims, ::metawarenn::Element::ElementType::kInt64, new_tensor_vec);
      graph_ptr->set_graph_initializers(pad_ip_tensor);
      graph_ptr->initializer_names_.insert(pad_ip_name);
      node_inputs[1] = pad_ip_name;
    }
    else if (op_type == kTfLiteBuiltinStridedSlice) {
      node_op_type = "Slice";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      /*const TfLiteStridedSliceParams* strided_slice_params = reinterpret_cast<const TfLiteStridedSliceParams*>(node->builtin_data);
      ::metawarenn::Attribute attr_begin_mask("begin_mask", std::vector<int64_t>{strided_slice_params->begin_mask});
      ::metawarenn::Attribute attr_ellipsis_mask("ellipsis_mask", std::vector<int64_t>{strided_slice_params->ellipsis_mask});
      ::metawarenn::Attribute attr_end_mask("end_mask", std::vector<int64_t>{strided_slice_params->end_mask});
      ::metawarenn::Attribute attr_new_axis_mask("new_axis_mask", std::vector<int64_t>{strided_slice_params->new_axis_mask});
      ::metawarenn::Attribute attr_shrink_axis_mask("shrink_axis_mask", std::vector<int64_t>{strided_slice_params->shrink_axis_mask});*/

      const int ip_tensor_id = node->inputs->data[0];
      const auto& ip_tensor = context->tensors[ip_tensor_id];
      std::vector<int> ip_dims_vec(ip_tensor.dims->data, ip_tensor.dims->data + ip_tensor.dims->size);

      node_inputs.pop_back(); //Removes the TFLite strides Input
      //Parse the Input 1(begin) & 2(end) value and do the CHW conversion along with int type handling
      //Begin Tensor Input
      std::string begin_ip_name = node_name + "_ip_begin";
      const int begin_tensor_id = node->inputs->data[1];
      const auto& begin_tensor = context->tensors[begin_tensor_id];
      std::vector<int> begin_dims_vec(begin_tensor.dims->data, begin_tensor.dims->data + begin_tensor.dims->size);
      auto num_elements_begin = std::accumulate(begin(begin_dims_vec), end(begin_dims_vec), 1, std::multiplies<int>());
      std::vector<int> tensor_begin(begin_tensor.data.i32, begin_tensor.data.i32 + num_elements_begin);

      std::vector<int64_t> begin_tensor_vec(num_elements_begin);
      begin_tensor_vec[0] = tensor_begin[0]; begin_tensor_vec[1] = tensor_begin[3];
      begin_tensor_vec[2] = tensor_begin[1]; begin_tensor_vec[3] = tensor_begin[2];

      ::metawarenn::Tensor begin_tensor_ip(begin_ip_name, std::vector<int>({begin_tensor_vec.size()}), ::metawarenn::Element::ElementType::kInt64, begin_tensor_vec);
      graph_ptr->set_graph_initializers(begin_tensor_ip);
      graph_ptr->initializer_names_.insert(begin_ip_name);
      node_inputs[1] = begin_ip_name; //Replace the existing unordered(HWC) float data with CHW int data

      //End Tensor Input
      std::string end_ip_name = node_name + "_ip_end";
      const int end_tensor_id = node->inputs->data[2];
      const auto& end_tensor = context->tensors[end_tensor_id];
      std::vector<int> end_dims_vec(end_tensor.dims->data, end_tensor.dims->data + end_tensor.dims->size);
      auto num_elements_end = std::accumulate(begin(end_dims_vec), end(end_dims_vec), 1, std::multiplies<int>());
      //std::vector<int> tensor_end(end_tensor.data.i32, end_tensor.data.i32 + num_elements_end);

      std::vector<int64_t> end_tensor_vec(num_elements_end);
      end_tensor_vec[0] = ip_dims_vec[0]; end_tensor_vec[1] = ip_dims_vec[3];
      end_tensor_vec[2] = ip_dims_vec[1]; end_tensor_vec[3] = ip_dims_vec[2];

      ::metawarenn::Tensor end_tensor_ip(end_ip_name, std::vector<int>({end_tensor_vec.size()}), ::metawarenn::Element::ElementType::kInt64, end_tensor_vec);
      graph_ptr->set_graph_initializers(end_tensor_ip);
      graph_ptr->initializer_names_.insert(end_ip_name);
      node_inputs[2] = end_ip_name; //Replace the existing unordered(HWC) float data with CHW int data
    }
    else if (op_type == kTfLiteBuiltinSqueeze) {
      node_op_type = "Squeeze";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteSqueezeParams* squeeze_params = reinterpret_cast<const TfLiteSqueezeParams*>(node->builtin_data);
      std::string squeeze_ip_name = node_name + "_ip";
      std::vector<int64_t> tensor_vec(squeeze_params->num_squeeze_dims, 0);
      for(int i=0; i<squeeze_params->num_squeeze_dims; i++)
          tensor_vec[i] = squeeze_params->squeeze_dims[i] + 1;//HWC to CHW handling for HW

      ::metawarenn::Tensor squeeze_tensor(squeeze_ip_name, std::vector<int>({tensor_vec.size()}), ::metawarenn::Element::ElementType::kInt64, tensor_vec);
      graph_ptr->set_graph_initializers(squeeze_tensor);
      graph_ptr->initializer_names_.insert(squeeze_ip_name);
      node_inputs.emplace_back(squeeze_ip_name); //Add Axes Input Tensor
    }
    else if (op_type == kTfLiteBuiltinReshape) {
      node_op_type = "Reshape";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteReshapeParams* reshape_params = reinterpret_cast<const TfLiteReshapeParams*>(node->builtin_data);
      std::string reshape_ip_name = node_name + "_ip";

      // Before adding Reshape's shape tensor, remove the initializer added before in initializer parsing.
      graph_ptr->remove_initializer_tensor(node_inputs[1]);
      graph_ptr->initializer_names_.erase(node_inputs[1]);

      // Adding Reshape input to avoid TFLite Int32 - ONNX expected Int64 datatype issue.
      std::vector<int64_t> tensor_vec(reshape_params->num_dimensions, 0);
      for(int i=0; i<reshape_params->num_dimensions; i++)
          tensor_vec[i] = reshape_params->shape[i];
      ::metawarenn::Tensor reshape_tensor(reshape_ip_name, std::vector<int>({tensor_vec.size()}), ::metawarenn::Element::ElementType::kInt64, tensor_vec);
      graph_ptr->set_graph_initializers(reshape_tensor);
      graph_ptr->initializer_names_.insert(reshape_ip_name);
      node_inputs[1] = reshape_ip_name; //Replace correct new_shape tensor(created from attributes)
    }
    else if (op_type == kTfLiteBuiltinSoftmax) {
      node_op_type = "Softmax";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      ::metawarenn::Attribute attr_axis("axis", (int64_t)1);//Defaults to 1(C) because, 0th axis mostly describes the batch_size(N)
      node_attributes.emplace_back(attr_axis);
    }
    else if (op_type == kTfLiteBuiltinHardSwish) {
      std::cout << "\n Convert TfLiteHardSwish to MwnnHardSwish\n";
      node_op_type = "HardSwish";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
    }
    else if (op_type == kTfLiteBuiltinMaximum) {
      node_op_type = "Max";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
    }
    else if (op_type == kTfLiteBuiltinSpaceToDepth) {
      std::cout << "\n Convert TfLiteSpaceToDepth to MwnnSpaceToDepth\n";
      node_op_type = "SpaceToDepth";// https://github.com/tensorflow/tensorflow/search?q=TfLiteDepthToSpaceParams
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteSpaceToDepthParams* space_to_depth_params = reinterpret_cast<const TfLiteSpaceToDepthParams*>(node->builtin_data);
      ::metawarenn::Attribute attr_block_size("block_size", std::vector<int64_t>{space_to_depth_params->block_size});
      node_attributes.emplace_back(attr_block_size);
    }
    else if (op_type == kTfLiteBuiltinArgMax) {
      std::cout << "\n Convert TfLiteArgMax to MwnnArgMax\n(Comment: the converter needs to read rediced_axis from tflite-input and write to MWNN-attribute)\n";
      node_op_type = "ArgMax";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      // To access a TFLite tensor, refer to
      // https://github.com/tensorflow/tensorflow/blob/5dcfc51118817f27fad5246812d83e5dccdc5f72/tensorflow/lite/c/common.h#L656
      // https://github.com/tensorflow/tensorflow/blob/5dcfc51118817f27fad5246812d83e5dccdc5f72/tensorflow/lite/c/common.h#L393
      const int axis_tensor_id = node->inputs->data[1]; // The axis to be reduced.
      const auto& axis_tensor = context->tensors[axis_tensor_id];
      if (axis_tensor.allocation_type != kTfLiteMmapRo) {
        std::cout << "\n model_builder.cc: ArgMax:axis is not kTfLiteMmapRo \n" << op_type;
        exit(-1);
      }
      std::vector<int> dims_vec(axis_tensor.dims->data, axis_tensor.dims->data + axis_tensor.dims->size);
      auto num_tensor_elements = std::accumulate(begin(dims_vec), end(dims_vec), 1, std::multiplies<int>());
      std::vector<int64_t> tensor_vec(axis_tensor.data.i32, axis_tensor.data.i32 + num_tensor_elements);
      ::metawarenn::Attribute attr_stride("axis", tensor_vec);
      node_attributes.emplace_back(attr_stride);
      node->inputs->size -= 1; // TFLite ArgMax{data, axis} -> ONNX ArgMax{data} + attr_axis.
      // TFLite always reduce that axis, so keepdims is always 0.
      ::metawarenn::Attribute attr_keepdims("keepdims", std::vector<int64_t>{0});
      node_attributes.emplace_back(attr_keepdims);
    }
    else if (op_type == kTfLiteBuiltinTransposeConv) {
      node_op_type = "ConvTranspose";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      // definition of transpose_conv_param
      // https://github.com/tensorflow/tensorflow/blob/783d5bea7effed1c38141752eb8ad395eeff2359/tensorflow/lite/c/builtin_op_data.h#L412
      const TfLiteTransposeConvParams* trans_conv_params = reinterpret_cast<const TfLiteTransposeConvParams*>(node->builtin_data);
      // (1) trans_conv_params->padding (2) trans_conv_params->stride_width (3) stride_height
      const int weight_tensor_id = node->inputs->data[1];
      const auto& weight_tensor = context->tensors[weight_tensor_id];
      // 1. tflite output_shape is input, write to attribute
      // auto_pad,,,output_padding,,pads,
      //group,kernel_shape, strides
      // dilations don't appear in tflite, default == 1 (in ONNX)
      // tflite treats `output_shape` as necessary node-input, extract later.
      std::string pad_mode;
      if (trans_conv_params->padding == kTfLitePaddingUnknown) {
        pad_mode = "NOTEST";
      }
      else if (trans_conv_params->padding == kTfLitePaddingSame) {
        pad_mode = "SAME_UPPER";
      }
      else if (trans_conv_params->padding == kTfLitePaddingValid) {
        pad_mode = "VALID";
      }
      else {
        std::cout << "\n Unsupported conv_transpose_padding: " << op_type;
        exit(1);
      }
      ::metawarenn::Attribute attr_auto_pad("auto_pad", std::vector<std::string>{pad_mode});
      node_attributes.emplace_back(attr_auto_pad);
      std::cout << "auto_pad: " << attr_auto_pad.get_string_data()[0] << ".\n";

      ::metawarenn::Attribute attr_dilate("dilations", std::vector<int64_t>{1,1});
      node_attributes.emplace_back(attr_dilate);
      ::metawarenn::Attribute attr_kernel_shape("kernel_shape", std::vector<int64_t>{weight_tensor.dims->data[1], weight_tensor.dims->data[2]});
      node_attributes.emplace_back(attr_kernel_shape);
      for (auto bbb: attr_kernel_shape.get_int_data()) {
        std::cout<<"attr_kernel_shape " << bbb<<" ~~~~\n";
      }
      ::metawarenn::Attribute attr_output_padding("output_padding", std::vector<int64_t>{0,0,0,0});
      node_attributes.emplace_back(attr_output_padding);
      // tflite treats output_shape as necessary node-input, extract
      const int output_shape_tensor_id = node->inputs->data[0];
      const auto& output_shape_tensor = context->tensors[output_shape_tensor_id];
      std::cout << "\n model_builder.cc: hello\n";
      if (output_shape_tensor.allocation_type != kTfLiteMmapRo) {
        std::cout << "\n model_builder.cc: " << op_type << ":output_shape is not kTfLiteMmapRo \n";
        exit(1);
      }
      std::cout << "\n model_builder.cc: hello2222\n";
      std::vector<int> dims_vec(output_shape_tensor.dims->data, output_shape_tensor.dims->data + output_shape_tensor.dims->size);
      auto num_tensor_elements = std::accumulate(begin(dims_vec), end(dims_vec), 1, std::multiplies<int>());
      std::cout << "\n model_builder.cc: hello333 " << num_tensor_elements <<".\n";
      std::vector<int64_t> tensor_vec(output_shape_tensor.data.i32, output_shape_tensor.data.i32 + num_tensor_elements);
      for(int z=0;z<tensor_vec.size();++z){ std::cout<<"output_shape ["<<z<<"]="<<tensor_vec[z]<<"\n"; }
      std::cout<<"!!!\n";
      ::metawarenn::Attribute attr_output_shape("output_shape", tensor_vec);
      node_attributes.emplace_back(attr_output_shape);
      // strides
      ::metawarenn::Attribute attr_stride("strides", std::vector<int64_t>{trans_conv_params->stride_height, trans_conv_params->stride_width});
      node_attributes.emplace_back(attr_stride);
      ::metawarenn::Attribute attr_pads("pads", std::vector<int64_t>()); // because PaddingSame is given, leave it empty.
      node_attributes.emplace_back(attr_pads);
      ::metawarenn::Attribute attr_group("group", std::vector<int64_t>{1});
      node_attributes.emplace_back(attr_group);
    }
    else if (op_type == kTfLiteBuiltinLogistic) {
      node_op_type = "Sigmoid";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
    }
    else if (op_type == kTfLiteBuiltinSum) {
      // It is `ReduceSum`, while kTfLiteBuiltinAdd is `Add`
      node_op_type = "ReduceSum";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteReducerParams* reduce_sum_params = reinterpret_cast<const TfLiteReducerParams*>(node->builtin_data);
      // keep_dims is `bool` in tflite; keepdims is `int` in onnx
      ::metawarenn::Attribute attr_keep_dims("keepdims",std::vector<int64_t> {(int64_t)reduce_sum_params->keep_dims});
      node_attributes.emplace_back(attr_keep_dims);
    }
    else if (op_type == kTfLiteBuiltinResizeBilinear) {
      node_op_type = "Resize";
      // `inputs` mismatch will be fixed in the afterwards swich-case section.
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteResizeNearestNeighborParams* resize_params = reinterpret_cast<const TfLiteResizeNearestNeighborParams*>(node->builtin_data);
      ::metawarenn::Attribute attr_mode("mode",std::vector<std::string> {std::string("linear")});
      node_attributes.emplace_back(attr_mode);
      // attributes: align_corners and half_pixel_centers.
      bool align_corners = resize_params->align_corners;
      bool half_pixel_centers = resize_params->half_pixel_centers;
      std::string coordinate_transformation_mode("half_pixel");
      if (half_pixel_centers) ;
      if (align_corners) coordinate_transformation_mode = "align_corners";
      ::metawarenn::Attribute attr_cord_trans_mode("coordinate_transformation_mode",std::vector<std::string> {coordinate_transformation_mode});
      node_attributes.emplace_back(attr_cord_trans_mode);
    }
    else if (op_type == kTfLiteBuiltinResizeNearestNeighbor) {
      node_op_type = "Resize";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteResizeNearestNeighborParams* resize_params = reinterpret_cast<const TfLiteResizeNearestNeighborParams*>(node->builtin_data);
      bool align_corners = resize_params->align_corners;
      bool half_pixel_centers = resize_params->half_pixel_centers;
      std::string coordinate_transformation_mode("half_pixel");
      if (half_pixel_centers) ;
      if (align_corners) coordinate_transformation_mode = "align_corners";
      ::metawarenn::Attribute attr_cord_trans_mode("coordinate_transformation_mode",std::vector<std::string> {coordinate_transformation_mode});
      node_attributes.emplace_back(attr_cord_trans_mode);
    }
    else if (op_type == kTfLiteBuiltinPrelu) {
      // the `slope` tensor is in input, no need to parse attribute
      node_op_type = "PRelu";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
    }
    else if (op_type == kTfLiteBuiltinSpaceToBatchNd) {
      node_op_type = "SpaceToBatch";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
    }
    else if (op_type == kTfLiteBuiltinDequantize) {
      // we currently support float16->float32 dequantization only
      // just store the tensor into initializers
      // merge `input tensor VALUE` and `output tensor NAME` to an INITIALIZER
      const int in_tensor_id = node->inputs->data[0];
      const auto& in_tensor = context->tensors[in_tensor_id];
      assert(in_tensor.allocation_type == kTfLiteMmapRo && "Only supports constant input yet");
      assert(in_tensor.type == kTfLiteFloat16 && "Only supports fp16 input yet");
      // extract number of elements (i.e. tensor size)
      /*std::vector<int> dims_vec = ParseTfLiteTensorDims(in_tensor);
      std::vector<float> vals_vec = ParseTfLiteTensorVals(in_tensor, dims_vec);
      const std::string fp32_name = context->tensors[node->outputs->data[0]].name;
      std::cout<<"Tensor name "<<fp32_name << " dims = ";
      for (auto _dim : dims_vec) {
        std::cout << _dim << " ";
      }
      std::cout << "! \n"<<std::flush;
      for (int _z = 0; _z < 5; ++_z) {
        std::cout << vals_vec[_z] <<" ";
      }
      std::cout << "! \n"<<std::flush;
      add_mwnn_initializer(graph_ptr, fp32_name,
                           get_mwnn_type_tf(in_tensor.type),
                           dims_vec, vals_vec);
      continue;*/
    }
    else {
      std::cout << "\n Unsupported op_type: " << op_type;
      exit(1);
    }
    /* Node's Fused Activation Handling
      - kTfLiteBuiltinDepthwiseConv2d
      - kTfLiteBuiltinConv2d
      - kTfLiteBuiltinAdd
    */
    if(!(activation_node_name.empty())) {
      std::vector<std::string> activation_node_inputs;
      std::vector<std::string> activation_node_outputs;
      std::vector<::metawarenn::Attribute> activation_node_attributes;
      std::cout << "\n activation_node_name : " << activation_node_name;
      std::cout << "\n activation_node_op_type : "<< activation_node_op_type;

      activation_node_inputs.emplace_back(node_name+"_output");
      activation_node_outputs.emplace_back(node_outputs[0]);
      node_outputs[0] = node_name+"_output";

      if(activation_node_op_type == "Clip") { //Relu6 Params & its Mapping to ONNX
        std::string clip_ip_min = activation_node_name + "_min";
        ::metawarenn::Tensor min_tensor(clip_ip_min, std::vector<int>({1}), ::metawarenn::Element::ElementType::kFloat, std::vector<float>({0}));
        graph_ptr->set_graph_initializers(min_tensor);
        graph_ptr->initializer_names_.insert(clip_ip_min);
        activation_node_inputs.emplace_back(clip_ip_min);

        std::string clip_ip_max = activation_node_name + "_max";
        ::metawarenn::Tensor max_tensor(clip_ip_max, std::vector<int>({1}), ::metawarenn::Element::ElementType::kFloat, std::vector<float>({6}));
        graph_ptr->set_graph_initializers(max_tensor);
        graph_ptr->initializer_names_.insert(clip_ip_max);
        activation_node_inputs.emplace_back(clip_ip_max);
      }
        CreateMWNNNode(graph_ptr, node_name, node_op_type, node_attributes, node_inputs, node_outputs);
        CreateMWNNNode(graph_ptr, activation_node_name, activation_node_op_type, activation_node_attributes, activation_node_inputs, activation_node_outputs);
      }
      else {
        CreateMWNNNode(graph_ptr, node_name, node_op_type, node_attributes, node_inputs, node_outputs);
      }
    }
  return graph_ptr;
}


/* TODO: High Level Graph to MetaWareNN Graph Representation,
         Apply Passes on MetaWareNN Graph,
         Generate Low Level Graph to run on devices*/
TfLiteStatus ModelBuilder::MetaWareNNCompile(std::shared_ptr<::metawarenn::Graph> graph) {
  std::cout << "\n In MetaWareNNCompile !!! ";
  static int subgraph_counter = 0;
  subgraph_counter++;
  ::metawarenn::optimizer::PassManager manager;
  if(HWC_TO_CHW)
  {
    for (auto g_t : graph->get_graph_initializers()) {
      if(g_t.get_dims().size() == 4) {
        //std::cout << "\n Name : " << g_t.get_name();
        for(auto node : graph->get_graph_nodes()) {
          if((node.get_op_type() == "Conv" && g_t.get_name() == node.get_inputs()[1]) or
             (node.get_op_type() == "DequantizeLinear" && g_t.get_name() == node.get_inputs()[0])) {
            //OHWI
            /*std::cout << "\t Dims : ";
            for (auto dim : g_t.get_dims())
              std::cout << dim << ",";*/
            ::metawarenn::optimizer::ConvertLayout cl(graph, g_t, 0, HWC_TO_CHW, 0, true);
            manager.RegisterPass(cl);
          }
        }
      }
      else {
        for (auto node : graph->get_graph_nodes()) {
          if(node.get_op_type() == "Mul" || node.get_op_type() == "Add") {
            for (auto n_ip : node.get_inputs()) {
              if(g_t.get_name() == n_ip) {
                std::cout << "\n Less Dimensiosna Name : " << g_t.get_name();
                std::cout << "\t Dims : ";
                for (auto dim : g_t.get_dims())
                  std::cout << dim << ",";
                ::metawarenn::optimizer::ExpandDimension ed(graph, g_t);
                manager.RegisterPass(ed);
              }
            }
          }
        }
      }
    }
      // (minxinx) TODO: validate this pass... if reshape receives 4D input, should we maybe
      // insert Transpose(NCHW2NHWC) ?
    for (auto g_t : graph->get_graph_ip_tensor()) {
      if(g_t.get_dims().size() == 4) {
        /*std::cout << "\n Name : " << g_t.get_name();
        std::cout << "\t Dims : ";
        for (auto dim : g_t.get_dims())
          std::cout << dim << ",";*/
        ::metawarenn::optimizer::ConvertLayout cl(graph, g_t, 0, HWC_TO_CHW, 0, false);
        manager.RegisterPass(cl);
      }
    }
  }
  auto node_list = graph->get_graph_nodes();
  for (int node_idx = 0; node_idx < graph->get_graph_nodes().size() ; node_idx++) {
    auto g_n = node_list[node_idx];
    /*if(g_n.get_op_type() == "Reshape") {
    if(g_n.get_op_type() == "Reshape") {
      ::metawarenn::optimizer::RemoveReshape rr(graph, g_n);
      std::cout << "\n MetaWareNNCC : " << rr.get_name();
      manager.RegisterPass(rr);
    }
    else if(g_n.get_op_type() == "Relu") {
      // This should be done in shared (tflite,onnxruntime,tvm,glow) MWNN Graph
      ::metawarenn::optimizer::FuseRelu fr(graph, g_n);
      std::cout << "\n MetaWareNNCC : " << fr.get_name();
      manager.RegisterPass(fr);
    }*/
  }
  /*::metawarenn::optimizer::CalculateOffset co(graph);
  manager.RegisterPass(co);*/
  manager.RunPasses();

  auto graph_ip_names = graph->get_graph_ip_names();
  for (auto g_n : graph->get_graph_nodes()) {
    for (auto n_ip : g_n.get_inputs()) {
      if(!(graph->initializer_names_.count(n_ip)) && !(std::count(graph_ip_names.begin(), graph_ip_names.end(), n_ip))) {
        if (graph->get_node_producers().count(n_ip)) {
          graph->set_node_consumer(n_ip, g_n.get_name());
        }
      }
    }
    for (auto n_op : g_n.get_outputs()) {
      graph->set_node_producer(n_op, g_n.get_name());
    }
  }
  for (auto itr : graph->get_node_producers()) {
    std::cout << "\n Produced Tensor : " << itr.first;
    std::cout << "\n      Producer Node : " << itr.second;
  }
  for (auto itr : graph->get_node_consumers()) {
    std::cout << "\n Consumed Tensor : " << itr.first;
    auto& vitr = itr.second;
    for (auto node_name : vitr) {
        std::cout << "\n      Consumer Node - " << node_name;
    }
  }

  #if INVOKE_NNAC
    std::cout << "\n ---------------------------Graph----------------------------- \n";
    std::cout << "\n Graph Name : " << graph->get_name();

    ::MWNN::MWNNGraphProto graph_proto;
    // Creates MWNNProto from MWNN Graph
    graph_proto = write_mwnn_proto(graph);
    std::cout << "\n Graph Name : " << graph->get_name();
    std::string name = graph->get_name();
    char* op_path = nullptr;
    op_path = getenv("NNAC_DUMPS_PATH");
    if(!IsPathExist(std::string(op_path))) {
      int check = mkdir(op_path, 0777);
      if(check != 0) {
        std::cout << "\nPlease check the directory path to store the serialized binary!!!!!";
        exit(1);
      }
    }
    auto proto_bin = std::string(op_path) + std::string(name) + ".bin";

    int fp = open(proto_bin.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    std::cout << fp;
    std::cout << graph_proto.SerializeToFileDescriptor(fp);
    close(fp);

    char* lib_path = nullptr;
    lib_path = getenv("METAWARENN_LIB_PATH");
    if(!IsPathExist(std::string(lib_path)))
      std::cout << "\nPlease check the MetaWareNN Library path!!!";
    std::cout << "\n\n=================Initiating NNAC python script via shell script======================\n";
    std::string cmd = "bash " + std::string(lib_path) +"/mwnnconvert/mwnn_convert.sh " + proto_bin + " " + op_path + " " + name + " " + std::to_string(subgraph_counter);
    const char *command = cmd.c_str();
    //system(command);
  #endif

  return kTfLiteOk;
  }

} // namespace metawarenn
} // namespace delegates
} // namespace tflite
