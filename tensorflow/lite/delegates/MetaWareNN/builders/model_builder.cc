#include "model_builder.h"
#include <cstdlib>
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
// helper function `GetTensorData` is defined in `tensor_ctypes.h`

namespace tflite {
std::vector<int> ParseTfLiteTensorDims(const TfLiteTensor &_tensor)
{
  std::vector<int> _vec(_tensor.dims->data, _tensor.dims->data + _tensor.dims->size);
  return _vec;
}

std::vector<float> ParseTfLiteTensorVals(const TfLiteTensor &_tensor,
                                         const std::vector<int> &_dims)
{
  const int _size = std::accumulate(std::begin(_dims), std::end(_dims), 1, std::multiplies<int>());
  // TfLitePtrUnion data has no size, use _dims from ParseTfLiteTensorDims
  // To get data, https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-tensorflow/blob/metawarenn_dev/tensorflow/lite/c/common.h#L310
  if (_tensor.type == kTfLiteFloat16) {
    /*typedef struct TfLiteFloat16 {
        uint16_t data;
      } TfLiteFloat16;*/
    const TfLiteFloat16 *_data = ::tflite::GetTensorData<TfLiteFloat16>(&_tensor);
    std::vector<float> _fp32;
    for (int i = 0; i < _size; ++i) {
      uint16_t d = _data[i].data;
      uint32_t _sign = d >> 15;  // 1 bit sign
      uint32_t _exp5 = (d >> 10) & 31; // 5 bits exponent
      uint32_t _frac = d & 1023; // 10 bits fraction, so 1 + 5 + 10 in 16 bits.
      float f = 1.0f;
      //if (i == 3) printf("u16=%d, _sign=%d, _exp=%d, _frac=%d\n", d, _sign, _exp5, _frac);
      if (_sign) f = -1.0f;
      if (_exp5 == 0) {
        if (_frac == 0) f = 0.0f;
        else f = f / 16384 * _frac / 1024; // (-1)^_sign * 2^-14 * (0.signiticantbits)_2
      }
      else { // (-1)^_sign * 2^(_exp-15) * (1.signiticantbits)_2
        float _signi = (float)(_frac + (1<<10)) / 1024;
        int _pot = -15 + _exp5;
        int _shift = 1;
        if (_pot >= 0) {
          _shift = 1 << _pot;
          f = f * _shift * _signi;
        }
        else {
          _shift = 1 << -_pot;
          f = f / _shift * _signi;
        }
        //if (i == 3) printf("(_frac+1<<10=%d),_signi=%f, _pot=%d, _shift=%d, f=%f\n", _frac+(1<<10),_signi, _pot, _shift, f);
      }
      _fp32.push_back(f);
      assert(_exp5 != 31 && "NaN fp16 encountered!");
    }
    /*for (int i=0;i<5;i++) std::cout<<_data[i].data<<" => " << _fp32[i] << " " << _fpdata[i] << "\n";
    std::cout<<"\n";*/
    return _fp32;
  }
  else if (_tensor.type == kTfLiteFloat32) {
    const float *_data = ::tflite::GetTensorData<float>(&_tensor);
    std::vector<float> _fp32(_data, _data + _size);
    for(int i=0;i<5;i++) printf("parsed v[%d] = %f\n", i, _fp32[i]);
    return _fp32;
  }
  else {
    std::cout<< "Not supported TfLiteTensor.type <"<< _tensor.type << ">.\n";
    exit(-27);
  }
}

void parse_inputs_outputs_names(const TfLiteContext *context,
                                const TfLiteNode *node, 
                                std::vector<std::string> &inputs,
                                std::vector<std::string> &outputs)
{
  inputs.clear(); outputs.clear();
  for (int i = 0; i < node->inputs->size; ++i) {
      const int tensor_id = node->inputs->data[i];
      inputs.emplace_back(context->tensors[tensor_id].name);
  }
  for (int i = 0; i < node->outputs->size; ++i) {
      const int tensor_id = node->outputs->data[i];
      outputs.emplace_back(context->tensors[tensor_id].name);
  }
  // return inputs, outputs
}

namespace delegates {
namespace metawarenn {
    
void add_mwnn_initializer(std::shared_ptr<::metawarenn::MWNNGraph> mwnn_graph,
                          const std::string &_name,
                          const ::metawarenn::ElementType::element_type &_type,
                          const std::vector<int> &_dims,
                          const std::vector<float> &_vals)
{
  ::metawarenn::MWNNTensor mwnn_tensor(_name, _dims, _type, _vals);
  mwnn_graph->set_graph_initializers(mwnn_tensor); // initializer_vector.emplace_back
  mwnn_graph->mwnn_initializer_names.insert(_name);
  // Question: why MCW::mwnn fake an extra constant node?
  // https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-tensorflow/blob/06562faf7b21244ddfbbd49dff94efb423ecb810/tensorflow/lite/delegates/MetaWareNN/builders/model_builder.cc#L352-L353
  std::cout<<_name<<"!!!!\n";
  //exit(-5);
}

void add_mwnn_node(std::shared_ptr<::metawarenn::MWNNGraph> mwnn_graph,
                   const std::string &_node_name,
                   const std::string &_op_type,
                   const std::vector<std::string> &_node_inputs,
                   const std::vector<std::string> &_node_outputs,
                   const std::vector<::metawarenn::MWNNAttribute> &_attrs)
{
  // MWNNNode(std::string m_name, std::string m_op_type, std::vector<MWNNAttribute> m_mwnn_attributes, std::vector<std::string> m_inputs,  std::vector<std::string> m_outputs);
  ::metawarenn::MWNNNode _node(_node_name, _op_type, _attrs, _node_inputs, _node_outputs);
  mwnn_graph->set_graph_nodes(_node); // the abstract node is stored in a vector
  // then the op_type-specific object is stored in map<string<NodeName>, shared_ptr<Node>>
  // currently only `optimizer/convert_layout.cc` uses it
  auto op_node = _node.get_node();
  mwnn_graph->mwnn_graph_nodes[_node.get_name()] = std::move(op_node);
  // trace: the MWNN-Node stores only high level info (op_type, mwnn_attributes, ...)
  // mwnn_attribute is wrapped info, get_node() creates specific `op_type` object,
  // i.e. mwnn_attributes("stride", {1,1}) is stored in ConvObject{member variable stride={1,1}}
}

void write_TFLiteTensor_into_MWNNTensor(std::shared_ptr<::metawarenn::MWNNGraph> mwnn_graph,
                                        const TfLiteTensor &_tensor)
{
  std::vector<int> dims_vec = ParseTfLiteTensorDims(_tensor);
  std::vector<float> vals_vec = ParseTfLiteTensorVals(_tensor, dims_vec);
  ::metawarenn::ElementType::element_type _type;
  _type = ::tflite::delegates::metawarenn::ModelBuilder::get_mwnn_type_tf(_tensor.type);
  add_mwnn_initializer(mwnn_graph, _tensor.name, _type,
                       dims_vec, vals_vec);
}

void convert_CHWN_to_NHWC(std::shared_ptr<::metawarenn::MWNNGraph> mwnn_graph, std::string initializer_name)
{
  // TODO: give `perm` as vector<int> as parameter to do Transpose
  std::cout<<"Converting DepthwiseConv weight IHWO to OHWI " << initializer_name << ".\n";
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

void convert_HWC_to_CHW(std::shared_ptr<::metawarenn::MWNNGraph> mwnn_graph, std::string initializer_name)
{
  std::cout<<"Converting PRelu slope HWC to CHW " << initializer_name << ".\n";
  auto init = mwnn_graph->get_initializer_tensor(initializer_name);
  auto dims = init.get_dims();
  std::vector<int> new_dims{dims[2], dims[0], dims[1]};
  auto buf = init.get_tensor(); // vector<float> tensor;
  std::vector<float> new_buf((new_dims[0]*new_dims[1]*new_dims[2]), 0);
  int channel = dims[2];
  int width = dims[0];
  int height = dims[1];
  for(int h = 0; h < height; ++h) {
    for(int w = 0; w < width; ++w) {
      for(int c = 0; c < channel; ++c) {
        new_buf[((c * height) + h) * width + w] = buf[((h * width) + w) * channel + c];
      }
    }
  }
  mwnn_graph->update_initializer_tensors(init.get_name(), new_dims, new_buf);
}

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
  mwnn_graph_ptr->set_graph_ip_names(input_tensor.name);
  //Fills Graph Input Tensor Details - Name, Dims
  ::metawarenn::MWNNTensor mwnn_ip_tensor(input_tensor.name, get_mwnn_type_tf(input_tensor.type), dims_ip_vec);
  auto ip_node = mwnn_ip_tensor.get_node();
  mwnn_graph_ptr->mwnn_graph_nodes[mwnn_ip_tensor.get_name()] = std::move(ip_node);
  mwnn_graph_ptr->set_graph_ip_tensor(mwnn_ip_tensor);

  //Set Graph Output Node
  context->GetNodeAndRegistration(context, subgraph_nodes_[subgraph_nodes_.size()-1], &node, &reg);
  tensor_id = node->outputs->data[0];
  const auto& output_tensor = context->tensors[tensor_id];
  std::vector<int> dims_op_vec(output_tensor.dims->data, output_tensor.dims->data + output_tensor.dims->size);
  mwnn_graph_ptr->set_graph_op_names(output_tensor.name);
  //Fills Graph Output Tensor Details - Name, Dims
  ::metawarenn::MWNNTensor mwnn_op_tensor(output_tensor.name, get_mwnn_type_tf(output_tensor.type), dims_op_vec);
  mwnn_graph_ptr->set_graph_op_tensor(mwnn_op_tensor);

  // If multiple layers share the constant tensor, we only keep one copy in MWNN
  //auto const_names = map<string, int>();
  std::map<std::string, int> const_names;

  for (size_t node_index = 0; node_index < subgraph_nodes_.size(); node_index++) {
    /*if (node_index > 3) {// 17 19v 20x
      break;
    }*/
    std::cout << "\n -------------------------------------------------------------------------------------------------------------";
    TfLiteNode* node;
    TfLiteRegistration* reg;
    const auto status = context->GetNodeAndRegistration(context, subgraph_nodes_[node_index], &node, &reg);
    auto op_type = reg->builtin_code;
    std::cout<<"\nhello "<<op_type<<" \n";

    std::string node_name;
    std::string node_op_type;
    std::vector<std::string> node_inputs;
    std::vector<std::string> node_outputs;
    std::vector<::metawarenn::MWNNAttribute> node_attributes;

    std::cout << "model_builder.cc parsing tflite op : " << op_type << ".\n";
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

      /*int activation_type;
      if(conv_params->activation == ::tflite::ActivationFunctionType_NONE)
        activation_type = ::metawarenn::ActivationType::Activation_None;
      else if(conv_params->activation == ::tflite::ActivationFunctionType_RELU)
        activation_type = ::metawarenn::ActivationType::Activation_Relu;
      else if(conv_params->activation == ::tflite::ActivationFunctionType_RELU6)
        activation_type = ::metawarenn::ActivationType::Activation_Relu6;

      ::metawarenn::MWNNAttribute mwnn_attr_activation("activation", std::vector<int>{activation_type});
      node_attributes.emplace_back(mwnn_attr_activation);*/
      std::string pad_mode = "NOTEST";
      ::metawarenn::MWNNAttribute mwnn_attr_auto_pad("auto_pad", std::vector<std::string>{pad_mode});
      node_attributes.emplace_back(mwnn_attr_auto_pad);
      std::cout << "auto_pad: " << mwnn_attr_auto_pad.get_string_data()[0] << ".\n";

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
      ::metawarenn::MWNNAttribute mwnn_attr_group("group", std::vector<int>{1});
      node_attributes.emplace_back(mwnn_attr_group);
    }
    else if (op_type == kTfLiteBuiltinDepthwiseConv2d) {
      // ONNX doesn't define DepthwiseConv, we should parse `group` from weight_shape.
      // By the way, the weight channel_order (IHWO) is different from Conv2D (OHWI),
      // at the end of this function we will transpose it to OHWI,
      // Then another function MetaWareNNCompile will align them to OIHW (for onnx)
      node_op_type = "Conv";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteDepthwiseConvParams* depthwise_conv_params = reinterpret_cast<const TfLiteDepthwiseConvParams*>(node->builtin_data);
      const int weight_tensor_id = node->inputs->data[1];
      const auto& weight_tensor = context->tensors[weight_tensor_id];
      // 1st: auto_pad (let's say, order from metawarenn_lib/op/conv.h)
      std::string pad_mode;
      if (depthwise_conv_params->padding == kTfLitePaddingUnknown) {
        pad_mode = "NOTEST";
        std::cout<<"DepthwiseConv padding need to be parsed.\n";
        /*::metawarenn::MWNNAttribute mwnn_attr_pad("pads", std::vector<int>{0, 0, 0, 0});
        node_attributes.emplace_back(mwnn_attr_pad);*/
        exit(-1);
      }
      else if (depthwise_conv_params->padding == kTfLitePaddingSame) {
        pad_mode = "SAME_UPPER";
      }
      else if (depthwise_conv_params->padding == kTfLitePaddingValid) {
        pad_mode = "VALID";
      }
      else {
        std::cout << "\n Unsupported depthwise_conv_padding: " << node_name;
        exit(1);
      }
      ::metawarenn::MWNNAttribute mwnn_attr_auto_pad("auto_pad", std::vector<std::string>{pad_mode});
      node_attributes.emplace_back(mwnn_attr_auto_pad);
      std::cout << "depthwiseconv auto_pad: " << mwnn_attr_auto_pad.get_string_data()[0] << ".\n";
      // 2nd: dilations
      ::metawarenn::MWNNAttribute mwnn_attr_dilate("dilations", std::vector<int>{depthwise_conv_params->dilation_height_factor, depthwise_conv_params->dilation_width_factor});
      node_attributes.emplace_back(mwnn_attr_dilate);
      // 3rd: group
      ::metawarenn::MWNNAttribute mwnn_attr_group("group", std::vector<int>{weight_tensor.dims->data[3]});
      node_attributes.emplace_back(mwnn_attr_group);
      // 4th: kernel_shape
      ::metawarenn::MWNNAttribute mwnn_attr_kernel_shape("kernel_shape", std::vector<int>{weight_tensor.dims->data[1], weight_tensor.dims->data[2]});
      node_attributes.emplace_back(mwnn_attr_kernel_shape);
      // 5th: pads
      ::metawarenn::MWNNAttribute mwnn_attr_pad("pads", std::vector<int>());
      node_attributes.emplace_back(mwnn_attr_pad);
      // 6th: strides
      ::metawarenn::MWNNAttribute mwnn_attr_stride("strides", std::vector<int>{depthwise_conv_params->stride_height, depthwise_conv_params->stride_width});
      node_attributes.emplace_back(mwnn_attr_stride);
      // let's some day write a general attribute_parser for kernel functions...
      // hope op_type_params could be passed-in by `auto`...
      // https://stackoverflow.com/questions/29944985/is-there-a-way-to-pass-auto-as-an-argument-in-c

      /*int activation_type;
      if(depthwise_conv_params->activation == ::tflite::ActivationFunctionType_NONE)
        activation_type = ::metawarenn::ActivationType::Activation_None;
      else if(depthwise_conv_params->activation == ::tflite::ActivationFunctionType_RELU)
        activation_type = ::metawarenn::ActivationType::Activation_Relu;
      else if(depthwise_conv_params->activation == ::tflite::ActivationFunctionType_RELU6)
        activation_type = ::metawarenn::ActivationType::Activation_Relu6;
      ::metawarenn::MWNNAttribute mwnn_attr_activation("activation", std::vector<int>{activation_type});
      node_attributes.emplace_back(mwnn_attr_activation);*/
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
      // TODO: it should be onnx `ReduceMean`,
      // current mwnn Mean follows tflite Mean, which is not expected.
      // So (1) read attribute `keepdims`, and (2) read input `axis` and write to attribute
      node_op_type = "ReduceMean";
      std::cout << "\nMap tflite Mean to onnx ReduceMean\n.";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteReducerParams* reduce_sum_params = reinterpret_cast<const TfLiteReducerParams*>(node->builtin_data);
      // tflite attr name: keep_dims, onnx attr name: keepdims
      ::metawarenn::MWNNAttribute mwnn_attr_keep_dims("keepdims",std::vector<int> {(int)reduce_sum_params->keep_dims});
      node_attributes.emplace_back(mwnn_attr_keep_dims);
      const int axes_tensor_id = node->inputs->data[1]; // The axis to be reduced.
      const auto& axes_tensor = context->tensors[axes_tensor_id];
      if (axes_tensor.allocation_type != kTfLiteMmapRo) {
        std::cout << "\n model_builder.cc: Mean:axes is not kTfLiteMmapRo \n" << op_type;
        exit(-1);
      }
      std::vector<int> dims_vec(axes_tensor.dims->data, axes_tensor.dims->data + axes_tensor.dims->size);
      auto num_tensor_elements = std::accumulate(begin(dims_vec), end(dims_vec), 1, std::multiplies<int>());
      std::vector<int> tensor_vec(axes_tensor.data.i32, axes_tensor.data.i32 + num_tensor_elements);
      for(int z=0;z<tensor_vec.size();++z){
        std::cout<<"reducemean axis ["<<z<<"]="<<tensor_vec[z]<<"\n";
      }
      std::cout<<"!!!\n";
      ::metawarenn::MWNNAttribute mwnn_attr_axes("axes", tensor_vec);
      node_attributes.emplace_back(mwnn_attr_axes);
      // The axes should be permuted from [1,2] to [2,3] when NHWC->NCHW
      // Is it in compile-phase?
      node->inputs->size -= 1; // TFLite Mean{data, axes} -> ONNX ReduceMean{data} + attr_axes.
    }
    else if (op_type == kTfLiteBuiltinFullyConnected) {
      /*node_op_type = "FullyConnected";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteFullyConnectedParams* fc_params = reinterpret_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);
      ::metawarenn::MWNNAttribute mwnn_attr_asymmetric_quantize_inputs("asymmetric_quantize_inputs",std::vector<int> {fc_params->asymmetric_quantize_inputs});
      node_attributes.emplace_back(mwnn_attr_asymmetric_quantize_inputs);
      ::metawarenn::MWNNAttribute mwnn_attr_keep_num_dims("keep_num_dims", std::vector<int>{fc_params->keep_num_dims});
      node_attributes.emplace_back(mwnn_attr_keep_num_dims);
      ::metawarenn::MWNNAttribute mwnn_attr_activation("activation", std::vector<int>{fc_params->activation});
      node_attributes.emplace_back(mwnn_attr_activation);
      ::metawarenn::MWNNAttribute mwnn_attr_weights_format("weights_format", std::vector<int>{fc_params->weights_format});
      node_attributes.emplace_back(mwnn_attr_weights_format);*/
      node_op_type = "Gemm";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      std::vector<std::string> inputs, outputs;
      parse_inputs_outputs_names(context, node, inputs, outputs);

      std::vector<int> input0_dims_vec = ParseTfLiteTensorDims(context->tensors[node->inputs->data[0]]);
      if (input0_dims_vec.size() > 2) {
        std::cout<< "Gemm input shape = ";
        for(auto d : input0_dims_vec) std::cout<<d<<" ";
        std::cout<<"\n";
        // insert Reshape node to 2D
        // mwnn_graph,_node_name,_op_type,_node_inputs,_node_outputs,_attrs
        //add_mwnn_node(mwnn_graph_ptr, "Reshape", );
        std::string rshp_name = inputs[0] + "/reshaped", param_name = rshp_name + "_param";
        
        add_mwnn_initializer(mwnn_graph_ptr, param_name,
                             ::metawarenn::ElementType::element_type::int64_,
                             std::vector<int>{2}, // dims_vec
                             std::vector<float>{input0_dims_vec[0], -1}); // vals_vec
        add_mwnn_node(mwnn_graph_ptr, rshp_name, "Reshape",
                      std::vector<std::string>{inputs[0], param_name},
                      std::vector<std::string>{rshp_name},
                      std::vector<::metawarenn::MWNNAttribute>());
        inputs[0] = rshp_name;
      }
      /*** write weight/bias into initializer ***/
      for(int i = 1; i <= 2; ++i) {
        const int in_tensor_id = node->inputs->data[i];
        const auto& in_tensor = context->tensors[in_tensor_id];
        assert(in_tensor.allocation_type == kTfLiteMmapRo && "FullyConnected->Gemm: weight/bias should be constant input");
        assert(in_tensor.type == kTfLiteFloat32 && "FullyConnected->Gemm: Only supports fp32 weight/bias yet");
        std::cout<<"\n tensor name = " << inputs[i] <<"\n";
        write_TFLiteTensor_into_MWNNTensor(mwnn_graph_ptr, in_tensor);
      }
      /*** parse attributes ***/
      const TfLiteFullyConnectedParams* fc_params = reinterpret_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);
      assert(fc_params->asymmetric_quantize_inputs == false); // bool
      assert(fc_params->keep_num_dims == false); // bool
      assert(fc_params->weights_format == 0); // TfLiteFullyConnectedWeightsFormat
      TfLiteFusedActivation activation = fc_params->activation;
      std::string act_out_name;
      if(activation != kTfLiteActNone) {
        // Will generate Gemm(input=inputs, output=[node_name]) +
        //    Activaction(input=[node_name], output=outputs)
        act_out_name = outputs[0];
        outputs[0] = node_name;
      }
      // gemm_attr = dict(alpha=1.0, beta=1.0, transA=0, transB=1)
      ::metawarenn::MWNNAttribute _transB("transB", std::vector<int>{1});
      node_attributes.emplace_back(_transB);
      /*** add node ***/
      add_mwnn_node(mwnn_graph_ptr, node_name, "Gemm",
                    inputs,
                    outputs,
                    node_attributes); // node_attributes is empty vector, or you can fill TransposeA/B, 
      
      if(activation != kTfLiteActNone) { // != 0
        std::string act_type = "None";
        std::vector<::metawarenn::MWNNAttribute> act_attrs;
        if (activation == kTfLiteActRelu) act_type = "Relu";
        else { // if (activation == kTfLiteActRelu6) act_type = "Clip";
          std::cout<< "\n Unsupported activation: " << activation << "\n";
          exit(-23);
        }
        add_mwnn_node(mwnn_graph_ptr, act_out_name, act_type,
              std::vector<std::string>{outputs[0]}, // input is output0 of Gemm
              std::vector<std::string>{act_out_name}, // output is FullyConnected output in TFLite
              act_attrs);
      }
      continue;
    }
    else if (op_type == kTfLiteBuiltinSplit) {
      node_op_type = "Split";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteSplitParams* split_params = reinterpret_cast<const TfLiteSplitParams*>(node->builtin_data);
      ::metawarenn::MWNNAttribute mwnn_attr_num_splits("num_splits  ", std::vector<int>{split_params->num_splits});
      node_attributes.emplace_back(mwnn_attr_num_splits);
    }
    else if (op_type == kTfLiteBuiltinPad) {
      // should get mode?
      node_op_type = "Pad";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      std::cout<< "Hits PAD !!!\n";
      std::string s("constant");
      ::metawarenn::MWNNAttribute mwnn_attr_mode("mode", std::vector<std::string>{s});
      node_attributes.emplace_back(mwnn_attr_mode);
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
      /*::metawarenn::MWNNAttribute mwnn_attr_strides("strides", std::vector<int>{strided_slice_params->steps});
      node_attributes.emplace_back(mwnn_attr_strides);*/
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
      /*const TfLiteReshapeParams* reshape_params = reinterpret_cast<const TfLiteReshapeParams*>(node->builtin_data);
      ::metawarenn::MWNNAttribute mwnn_attr_shape("shape", std::vector<int>{reshape_params->shape[0], reshape_params->shape[1]});
      node_attributes.emplace_back(mwnn_attr_shape);*/
      // TFLite has out_shape in both attribute and input
      // let's just take the info in `input`.
    }
    else if (op_type == kTfLiteBuiltinSoftmax) {
      node_op_type = "Softmax";
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteSoftmaxParams* softmax_params = reinterpret_cast<const TfLiteSoftmaxParams*>(node->builtin_data);
      int beta = (int32_t)softmax_params->beta;
      if (beta != 1) {
        std::cout << "\nReceived TfLiteSoftmax beta!=1 \n";
        exit(-1);
      }
      // axis is not given in tflite; both tflite and onnx takes -1 as default.
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
      ::metawarenn::MWNNAttribute mwnn_attr_block_size("block_size", std::vector<int>{space_to_depth_params->block_size});
      node_attributes.emplace_back(mwnn_attr_block_size);
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
      std::vector<int> tensor_vec(axis_tensor.data.i32, axis_tensor.data.i32 + num_tensor_elements);
      ::metawarenn::MWNNAttribute mwnn_attr_stride("axis", tensor_vec);
      node_attributes.emplace_back(mwnn_attr_stride);
      node->inputs->size -= 1; // TFLite ArgMax{data, axis} -> ONNX ArgMax{data} + attr_axis.
      // TFLite always reduce that axis, so keepdims is always 0.
      ::metawarenn::MWNNAttribute mwnn_attr_keepdims("keepdims", std::vector<int>{0});
      node_attributes.emplace_back(mwnn_attr_keepdims);
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
      ::metawarenn::MWNNAttribute mwnn_attr_auto_pad("auto_pad", std::vector<std::string>{pad_mode});
      node_attributes.emplace_back(mwnn_attr_auto_pad);
      std::cout << "auto_pad: " << mwnn_attr_auto_pad.get_string_data()[0] << ".\n";

      ::metawarenn::MWNNAttribute mwnn_attr_dilate("dilations", std::vector<int>{1,1});
      node_attributes.emplace_back(mwnn_attr_dilate);
      ::metawarenn::MWNNAttribute mwnn_attr_kernel_shape("kernel_shape", std::vector<int>{weight_tensor.dims->data[1], weight_tensor.dims->data[2]});
      node_attributes.emplace_back(mwnn_attr_kernel_shape);
      for (auto bbb: mwnn_attr_kernel_shape.get_int_data()) {
        std::cout<<"mwnn_attr_kernel_shape " << bbb<<" ~~~~\n";
      }
      ::metawarenn::MWNNAttribute mwnn_attr_output_padding("output_padding", std::vector<int>{0,0,0,0});
      node_attributes.emplace_back(mwnn_attr_output_padding);
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
      std::vector<int> tensor_vec(output_shape_tensor.data.i32, output_shape_tensor.data.i32 + num_tensor_elements); 
      for(int z=0;z<tensor_vec.size();++z){ std::cout<<"output_shape ["<<z<<"]="<<tensor_vec[z]<<"\n"; }
      std::cout<<"!!!\n";
      ::metawarenn::MWNNAttribute mwnn_attr_output_shape("output_shape", tensor_vec);
      node_attributes.emplace_back(mwnn_attr_output_shape);
      // strides
      ::metawarenn::MWNNAttribute mwnn_attr_stride("strides", std::vector<int>{trans_conv_params->stride_height, trans_conv_params->stride_width});
      node_attributes.emplace_back(mwnn_attr_stride);
      ::metawarenn::MWNNAttribute mwnn_attr_pads("pads", std::vector<int>()); // because PaddingSame is given, leave it empty.
      node_attributes.emplace_back(mwnn_attr_pads);
      ::metawarenn::MWNNAttribute mwnn_attr_group("group", std::vector<int>{1});
      node_attributes.emplace_back(mwnn_attr_group);
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
      ::metawarenn::MWNNAttribute mwnn_attr_keep_dims("keepdims",std::vector<int> {(int)reduce_sum_params->keep_dims});
      node_attributes.emplace_back(mwnn_attr_keep_dims);
    }
    else if (op_type == kTfLiteBuiltinResizeBilinear) {
      node_op_type = "Resize";
      // `inputs` mismatch will be fixed in the afterwards swich-case section.
      node_name = node_op_type + std::to_string(subgraph_nodes_[node_index]);
      const TfLiteResizeNearestNeighborParams* resize_params = reinterpret_cast<const TfLiteResizeNearestNeighborParams*>(node->builtin_data);
      ::metawarenn::MWNNAttribute mwnn_attr_mode("mode",std::vector<std::string> {std::string("linear")});
      node_attributes.emplace_back(mwnn_attr_mode);
      // attributes: align_corners and half_pixel_centers.
      bool align_corners = resize_params->align_corners;
      bool half_pixel_centers = resize_params->half_pixel_centers;
      std::string coordinate_transformation_mode("half_pixel");
      if (half_pixel_centers) ;
      if (align_corners) coordinate_transformation_mode = "align_corners";
      ::metawarenn::MWNNAttribute mwnn_attr_cord_trans_mode("coordinate_transformation_mode",std::vector<std::string> {coordinate_transformation_mode});
      node_attributes.emplace_back(mwnn_attr_cord_trans_mode);
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
      ::metawarenn::MWNNAttribute mwnn_attr_cord_trans_mode("coordinate_transformation_mode",std::vector<std::string> {coordinate_transformation_mode});
      node_attributes.emplace_back(mwnn_attr_cord_trans_mode);
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
      std::vector<int> dims_vec = ParseTfLiteTensorDims(in_tensor);
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
      add_mwnn_initializer(mwnn_graph_ptr, fp32_name, 
                           get_mwnn_type_tf(in_tensor.type),
                           dims_vec, vals_vec);
      
      continue;
    }
    else {
      std::cout << "\n Unsupported op_type: " << op_type;
      std::cout << " in model_builder.cc\n";
      exit(1);
    }

    // Converting TFLite Node inputs to ONNX
    for (int i = 0; i < node->inputs->size; ++i) {
      const int tensor_id = node->inputs->data[i];
      node_inputs.emplace_back(context->tensors[tensor_id].name);
    }
    // Align the tflite inputs order/definition to onnx
    switch (op_type) {
      case kTfLiteBuiltinTransposeConv: {
        // output_shape, weight, data -> data, weight
        std::swap(node_inputs[0], node_inputs[2]);
        node_inputs.pop_back();
        std::cout << "\nConvTranspose inputs: " << node_inputs[0] << node_inputs[1] << " hello\n";
        break;
      }
      case kTfLiteBuiltinResizeBilinear:
      case kTfLiteBuiltinResizeNearestNeighbor:
        // tflite{X,sizes} -> onnx{X, roi, scales, sizes}
        node_inputs.insert(node_inputs.begin()+1, "");
        node_inputs.insert(node_inputs.begin()+2, "");
        std::cout << "\nResize inputs: ";
        for(int k=0;k<node_inputs.size();k++) std::cout<< " !"<< node_inputs[k] << "! ";
        std::cout<< "\n";
      default:
        break;
    }

    // Converting TFLite Node outputs to ONNX
    for (int i = 0; i < node->outputs->size; ++i) {
      const int tensor_id = node->outputs->data[i];
      node_outputs.emplace_back(context->tensors[tensor_id].name);
    }
    for (int i = 0; i < node->inputs->size; ++i) {
      const int tensor_id = node->inputs->data[i];
      const auto& input_tensor = context->tensors[tensor_id];
      if (input_tensor.allocation_type == kTfLiteMmapRo) {
          const std::string tensor_name = input_tensor.name;
          // 1. If the constant tensor is absent in ONNX node-inputs definition, skip
          if (node_inputs.end() == std::find(node_inputs.begin(), node_inputs.end(), tensor_name)) {
            continue;
          }
          // 2. If the constant tensor is shared with previous layers (so already added), skip
          if (const_names.end() != const_names.find(tensor_name)) {
            // shared constant tensor
            std::cout << "Tensor " << tensor_name << "is shared!!!\n" << std::flush;
            continue;
          }

          const_names[tensor_name] = 1;
          
          std::vector<int> dims_vec(input_tensor.dims->data, input_tensor.dims->data + input_tensor.dims->size);
          auto num_tensor_elements = std::accumulate(begin(dims_vec), end(dims_vec), 1, std::multiplies<int>());
          std::vector<float> tensor_vec;
          if (input_tensor.type == kTfLiteInt32) {
            tensor_vec = std::vector<float>(input_tensor.data.i32, input_tensor.data.i32 + num_tensor_elements);
            std::cout << "Tensor " << tensor_name << "[:] = ";
            for (auto z : tensor_vec) std::cout<< " " << z;
            std::cout<<"\n check_int\n"<<std::flush;
          }
          else {
            tensor_vec = std::vector<float>(input_tensor.data.f, input_tensor.data.f + num_tensor_elements);
          }
          /*if (op_type == kTfLiteBuiltinReshape) {
            std::cout << "Reshape input !!! \n";
            for (auto v : tensor_vec) std::cout << v << ",int( " << int(v)<<".\n";
            std::cout << "\n";
          }else if (op_type == kTfLiteBuiltinPad) {
            std::cout << "Pad input !!! \n";
            for (auto v : tensor_vec) std::cout << v << ",int( " << int(v)<<".\n";
            std::cout << "\n";
          }*/
          ::metawarenn::MWNNTensor mwnn_tensor(input_tensor.name, dims_vec, get_mwnn_type_tf(input_tensor.type), tensor_vec);
          mwnn_graph_ptr->set_graph_initializers(mwnn_tensor);

          auto const_node = mwnn_tensor.get_constant_node();
          mwnn_graph_ptr->mwnn_graph_nodes[mwnn_tensor.get_name()] = std::move(const_node);
          std::cout<< "mwnn_tensor index =  "<<mwnn_tensor.get_index()<<"\n";

          mwnn_graph_ptr->mwnn_initializer_names.insert(input_tensor.name);
      }
    }
    ::metawarenn::MWNNNode mwnn_node(node_name, node_op_type, node_attributes, node_inputs, node_outputs);
    mwnn_graph_ptr->set_graph_nodes(mwnn_node);
    auto op_node = mwnn_node.get_node();
    mwnn_graph_ptr->mwnn_graph_nodes[mwnn_node.get_name()] = std::move(op_node);
    // align DepthwiseConv2d weight from IHWO to OHWI
    if (op_type == kTfLiteBuiltinDepthwiseConv2d) {
        // int _id = node->inputs->data[1];
        // std::string W_name = context->tensors[_id].name;
        std::cout<<"\nHELLO transpose depthwise2d\n";
        convert_CHWN_to_NHWC(mwnn_graph_ptr, node_inputs[1]);
        //exit(9);
    }
    else if (op_type == kTfLiteBuiltinPrelu) {
      // tflite 1x1xC -> onnx Cx1x1
      std::cout<<"\nHELLO transpose Prelu\n";
      convert_HWC_to_CHW(mwnn_graph_ptr, node_inputs[1]);
    }
    /*else if (op_type == kTfLiteBuiltinTransposeConv) {
        // tflite OHWI -> onnx IOHW will be handled in compile phase
    }*/
  }
  //exit(-1);
  return mwnn_graph_ptr;
}

/* TODO: High Level Graph to MetaWareNN Graph Representation,
         Apply Passes on MetaWareNN Graph,
         Generate Low Level Graph to run on devices*/
TfLiteStatus ModelBuilder::MetaWareNNCompile(std::shared_ptr<::metawarenn::MWNNGraph> mwnn_graph) {
  std::cout << "\n In MetaWareNNCompile !!! ";
  static int subgraph_counter = 0;
  subgraph_counter++;
  //Call Passes
  // 1st goal: align convolutional weight data-format
  // onnx: OIHW, Conv2D: OHWI, DepthwiseConv2D: IHWO
  // In BuildGraph let's map DepthwiseConv2D to Conv2D (so transpose there),
  // and both will be Conv here. 
  ::metawarenn::optimizer::PassManager manager;
  /*for (auto node : mwnn_graph->get_graph_nodes())
  {
    //Convert weight layout to common NHWC format before passes
    if(node.get_op_type() == "DepthwiseConv")// Conv && group!=1
    {
      convert_CHWN_to_NHWC(mwnn_graph, node.get_inputs()[1]);
    }
  }*/
  if(HWC_TO_CHW)
  { 
    for (auto g_t : mwnn_graph->get_graph_initializers()) {
      if(g_t.get_dims().size() == 4) {
        // (minxinx) TODO: validate this pass... on Conv-weights, Squeeze-axes, Resize-sizes, ...
        //std::cout << "\n Name : " << g_t.get_name();
        for(auto node: mwnn_graph->get_graph_nodes()) {
          if(node.get_op_type() == "Conv" && g_t.get_name() == node.get_inputs()[1]) {
            // OHWI
            /*std::cout << "\t Dims : ";
            for (auto dim : g_t.get_dims())
              std::cout << dim << ",";*/
            ::metawarenn::optimizer::ConvertLayout cl(mwnn_graph, g_t, 0, HWC_TO_CHW, true);
            manager.register_pass(cl);
          }
        }
      }
      // else if init.get_dims() == 4D, we transpose PRelu's `slope` tensor?
      // we currently do it in BuildGraph phase...
    }
    for (auto g_t : mwnn_graph->get_graph_ip_tensor()) {
      // (minxinx) TODO: validate this pass... if reshape receives 4D input, should we maybe
      // insert Transpose(NCHW2NHWC) ?
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
    /*if(g_n.get_op_type() == "Reshape") {
      ::metawarenn::optimizer::RemoveReshape rr(mwnn_graph, g_n);
      std::cout << "\n MetaWareNNCC : " << rr.get_name();
      manager.register_pass(rr);
    }
    else if(g_n.get_op_type() == "Relu") {
      // This should be done in shared (tflite,onnxruntime,tvm,glow) MWNN Graph
      ::metawarenn::optimizer::FuseRelu fr(mwnn_graph, g_n);
      std::cout << "\n MetaWareNNCC : " << fr.get_name();
      manager.register_pass(fr);
    }*/
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
