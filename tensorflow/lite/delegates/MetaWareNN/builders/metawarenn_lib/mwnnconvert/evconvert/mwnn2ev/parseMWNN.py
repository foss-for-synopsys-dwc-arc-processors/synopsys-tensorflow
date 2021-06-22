#!/usr/bin/env python
# ============================================================================= 
# Copyright 2020  Synopsys, Inc.
# This file and the associated documentation are proprietary to Synopsys, 
# Inc., and may only be used in accordance with the terms and conditions of 
# a written license agreement with Synopsys, Inc.
# Notwithstanding contrary terms in the DFPUC, Licensee may provide the
# binaries of the EV Runtime and Utilities Option to its end-customer that
# purchase Licensee ICs that incorporate the Synopsys EV processor core,
# subject to confidentiality terms no less restrictive than those contained in
# the DFPUC.  All other use, reproduction, or distribution of this file
# is strictly prohibited.
# =============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import OrderedDict
from onnx import numpy_helper
import numpy as np

class ConverterRegistry(object):
    registry_ = {}
    caffe_network = OrderedDict()

    CAFFE_NAME_SLASH = '_002f_'
    CAFFE_NAME_DOT = '_002e_'
    CAFFE_NAME_AT = '_0040_'
    CAFFE_NAME_DASH = '_002g_'
    CAFFE_NAME_COLON = '_002h_'

    model_name = ''
    dynamic_reshape_dict = None
    dynamic_value_dict = None
    user_input_shape = []
    caffe_first_nodes = []

    def __init__(self, mwnn_file, graphproto, input_shape=[], first_nodes=[]):
        ConverterRegistry.model_name = mwnn_file
        ConverterRegistry.dynamic_reshape_dict = None
        ConverterRegistry.dynamic_value_dict = None
        ConverterRegistry.user_input_shape = input_shape
        ConverterRegistry.caffe_first_nodes.clear()

        ConverterRegistry.caffe_network.clear()
        caffe_network = ConverterRegistry.caffe_network
        # to keep the caffe input order always the same as original onnx model
        data1 = []
        if first_nodes == []:
            data1 = [n.name for n in graphproto.input]
        else:
            for f in first_nodes:
                data1.append(f)
        data2 = [n.name for n in graphproto.initializer]
        data_all = [i for i in data1 if i not in data2]

        #data_all = (set([n.name for n in model.graph.input]) -
        #            set([n.name for n in model.graph.initializer]))
        for t in range(len(data_all)):
            data = data_all[t]
            if first_nodes == []:
                print("inside if")
                data = [n for n in graphproto.input if n.name == data][0]
            else:
                print("first_nodes: ", first_nodes)
                data = [n for n in graphproto.node if n.name == data][0]
            caffe_network[data.name] = {}
            if first_nodes == []:
                caffe_network[data.name]['name'] = ConverterRegistry.caffe_layer_name(data.name)
            else:
                caffe_network[data.name]['name'] = ConverterRegistry.caffe_layer_name(data.input[0])
                ConverterRegistry.caffe_first_nodes.append(data.input[0])
            print(data_all)
            print("data: " , data)
            shape = []
            if first_nodes == []:
                for dim in data.dims:
                    shape.append(dim)
                print("shape: ", shape)

            caffe_network[data.name]['functions'] = "{} = L.Input(shape=dict(dim={}))\n".format(
                caffe_network[data.name]['name'], shape)
            caffe_network[data.name]['output_name'] = caffe_network[data.name]['name']

    @classmethod
    def Register(cls, op_name):
        """A decorator for registering operators mappings."""

        def Wrapper(func):
            cls.registry_[op_name] = func
            return func

        return Wrapper

    @classmethod
    def ConverterLayer(cls, model, node, node_name, inputs, outputs, **kwargs):
        try:
            # Handles Conv, fused Conv+Relu
            if(node.op_type == "Conv"):
                activation_valid = False
                for attr in node.attribute:
                    # Checks for valid activation
                    if(attr.name == "activation" and attr.ints[0] != 0):
                        activation_valid = True
                        conv_output = []
                        conv_output.append(node_name+"_out")
                        cls.caffe_network[conv_output[0]] = cls.registry_[node.op_type](
                            model, node, node_name, inputs, conv_output, **kwargs)
                        relu_node_name = node_name+"_relu"
                        relu_inputs = []
                        relu_outputs = []
                        relu_inputs.append(conv_output[0])
                        relu_outputs.append(outputs[0])
                        caffe_layer = init_node(model, relu_node_name, relu_inputs, relu_outputs)
                        # Adds separate caffe layer for Relu
                        if(attr.ints[0] == 1):
                            caffe_layer['functions'] = '{output_name} = L.ReLU({input_names})\n'.format(
                            **caffe_layer)
                        # Adds separate caffe layer for Relu6
                        elif(attr.ints[0] == 2):
                            caffe_layer['functions'] = '{output_name} = L.ReLU({input_names}relu6=True)\n'.format(
                                **caffe_layer)
                        cls.caffe_network[outputs[0]] = caffe_layer
                        print(cls.caffe_network[conv_output[0]]['functions'])
                # If no valid activation, then add caffe layer for Conv node
                if(not activation_valid):
                    cls.caffe_network[outputs[0]] = cls.registry_[node.op_type](
                        model, node, node_name, inputs, outputs, **kwargs)
            # Handles Op types other than Relu
            else:
                cls.caffe_network[outputs[0]] = cls.registry_[node.op_type](
                    model, node, node_name, inputs, outputs, **kwargs)
            print(cls.caffe_network[outputs[0]]['functions'])
        except KeyError as err:
            # raise KeyError('No translator registered for layer: %s yet.' %
            #               str(layer))
            print('WARNING: lack parsing of', node_name,"-", node.op_type, "- might casue ERROR in conversion!!!\n")
            # raise err

    @classmethod
    def caffe_layer_name(cls, layer_name):
        layer_name = "layer_" + layer_name
        layer_name = layer_name.replace(":", cls.CAFFE_NAME_COLON)
        layer_name = layer_name.replace("-", cls.CAFFE_NAME_DASH)
        layer_name = layer_name.replace("@", cls.CAFFE_NAME_AT)
        layer_name = layer_name.replace(".", cls.CAFFE_NAME_DOT)
        layer_name = layer_name.replace("/", cls.CAFFE_NAME_SLASH)
        return 'nn.' + layer_name


# Each node should return a dict with items:
# name: the Caffe layer name
# input_names: the input layer name(s)
# output_name: usually it is "name", sometimes it isn't due to an operation
#              of ONNX maybe produce two or more Caffe layers.
# functions: the current ONNX node's corresponding Caffe functions
def init_node(graphproto, node_name, inputs, outputs):
    caffe_layer = {}
    caffe_layer['name'] = ConverterRegistry.caffe_layer_name(outputs[0])

    caffe_layer['input_names'] = ''
    for n in inputs:
        if ConverterRegistry.caffe_network.get(n):
            caffe_layer['input_names'] += ConverterRegistry.caffe_network[n]['output_name'] + ', '
    if caffe_layer['input_names'].find(', , '):
        caffe_layer['input_names'] = caffe_layer['input_names'].replace(', , ', ', ')
    if caffe_layer['input_names'] == '' and len(inputs) > 0:
        caffe_layer['input_names'] = ConverterRegistry.caffe_layer_name(inputs[0]) + ', '
    caffe_layer['output_name'] = caffe_layer['name']
    return caffe_layer


@ConverterRegistry.Register('Conv')
def parse_Conv(graphproto, node, node_name, inputs, outputs):
    caffe_layer = init_node(graphproto, node_name, inputs, outputs)

    weight = node.input[1]

    if len(inputs) == 1:
        # 2nd input (initializer) specifies the weights
        caffe_layer['weight'] = [w for w in graphproto.initializer
                                 if w.name == weight][0]
        num_output = caffe_layer['weight'].dims[0]
    else:
        # 2nd input (constant) specifies the weights
        caffe_layer['weight'] = [w for w in graphproto.node
                                 if w.output[0] == inputs[1]][0]
        if caffe_layer['weight'].op_type != "Constant":
            # special case in mlperf_tvm models
            if caffe_layer['weight'].op_type == 'Transpose':
                for attr in caffe_layer['weight'].attribute:
                    if attr.name == 'perm':
                        perm = [int(n) for n in attr.ints]
                weight = caffe_layer['weight'].input[0]
                weight_tensor = [n for n in graphproto.initializer
                                if n.name == weight]
                if weight_tensor != []:
                    shape = [n for n in weight_tensor[0].dims]
                    caffe_layer['weight'] = weight_tensor[0].float_data
                    if caffe_layer['weight'] == []:  # raw_data format
                        caffe_layer['weight'] = numpy_helper.to_array(weight_tensor[0])
                else:
                    caffe_layer['weight'] = [w for w in graphproto.node
                                             if w.output[0] == weight][0]
                    shape = [n for n in caffe_layer['weight'].attribute[0].t.dims]
                    caffe_layer['weight'] = caffe_layer['weight'].attribute[0].t
                    caffe_layer['weight'] = numpy_helper.to_array(caffe_layer['weight'])
                caffe_layer['weight'] = np.transpose(np.array(caffe_layer['weight']).reshape(shape), perm)
                num_output = caffe_layer['weight'].shape[0]
                caffe_layer['weight'] = numpy_helper.from_array(caffe_layer['weight'])
            else:
                # hard coded hack for SiameseRPN
                if node_name == '52':
                    caffe_layer['input_names'] = caffe_layer['input_names'][:caffe_layer['input_names'].find(",") + 2]
                    num_output = 10
                    caffe_layer['weight'] = numpy_helper.from_array(np.load('/home/yche/Downloads/cconv_weight.npy'))
                elif node_name == '53':
                    caffe_layer['input_names'] = caffe_layer['input_names'][:caffe_layer['input_names'].find(",") + 2]
                    num_output = 20
                    caffe_layer['weight'] = numpy_helper.from_array(np.load('/home/yche/Downloads/rconv_weight.npy'))
                else:
                    print("Dynamic error: Conv weight for layer {} is not constant!".format(node_name))
                    quit(1)
        else:
            num_output = caffe_layer['weight'].attribute[0].t.dims[0]
            caffe_layer['weight'] = caffe_layer['weight'].attribute[0].t

    if len(node.input) > 2:
        if len(inputs) == 1:
            # 3rd input (initializer) specifies the bias
            caffe_layer['bias'] = [b for b in graphproto.initializer
                                   if b.name == node.input[2]][0]
        else:
            # 3rd input (constant) specifies the bias
            caffe_layer['bias'] = [b for b in graphproto.node
                                   if b.output[0] == inputs[2]][0]
            caffe_layer['input_names'] = caffe_layer['input_names'].replace(', , ', ', ')
            caffe_layer['bias'] = caffe_layer['bias'].attribute[0].t

    stride_h = stride_w = 1
    kernel_h = kernel_w = 1
    # for not 2D case
    kernel_shape = 0
    pad_l = pad_r = pad_t = pad_b = 0
    group = 1
    dilations = [1, 1]
    for attr in node.attribute:
        if attr.name == 'kernel_shape':
            if len(attr.ints) == 1:
                kernel_shape = attr.ints[0]
            elif len(attr.ints) > 2:
                kernel_shape = [int(n) for n in attr.ints]
            else:
                [kernel_h, kernel_w] = attr.ints[:2]
        if attr.name == 'strides':
            if len(attr.ints) == 1:
                stride = attr.ints[0]
            elif len(attr.ints) > 2:
                stride = [int(n) for n in attr.ints]
            else:
                [stride_h, stride_w] = attr.ints[:2]
        if attr.name == 'pads':
            if len(attr.ints) == 2:
                # TODO: no support for 1D asymmetric paddding in Caffe yet
                pad = attr.ints[0]
            elif len(attr.ints) > 4:
                # TODO: no support for non-2D asymmetric paddding in Caffe yet
                # FIXME: add support for "once per spatial dimension" case
                pad = attr.ints[0]
            else:
                [pad_t, pad_l, pad_b, pad_r] = attr.ints
        if attr.name == 'group':
            group = attr.ints[0]
        if attr.name == "dilations":
            dilations = [int(n) for n in attr.ints]

    caffe_layer['functions'] = "{output_name} = L.Convolution({input_names}".format(**caffe_layer)
    # for non-2D conv
    if kernel_shape != 0:
        caffe_layer['functions'] += "num_output={}, pad={}, kernel_size={}, " \
                                    "stride={}, group={}, dilation={}".format(
            num_output, pad, kernel_shape, stride, group, dilations)
    else:
        caffe_layer['functions'] += "num_output={}, pad_t={}, pad_b={}, pad_l={}, pad_r={}, kernel_h={}, kernel_w={}, " \
                                    "stride_h={}, stride_w={}, group={}, dilation={}".format(
        num_output, pad_t, pad_b, pad_l, pad_r,
        kernel_h, kernel_w, stride_h, stride_w, group, dilations)
    if not caffe_layer.get('bias'):
        caffe_layer['functions'] += ', bias_term=False'

    caffe_layer['functions'] += ")\n"

    return caffe_layer

@ConverterRegistry.Register('Relu')
def parse_Relu(model, node, node_name, inputs, outputs):
    caffe_layer = init_node(model, node_name, inputs, outputs)
    caffe_layer['functions'] = '{output_name} = L.ReLU({input_names})\n'.format(
        **caffe_layer)

    return caffe_layer

# GlobalAveragePool consumes an input tensor X and applies average
# pooling across the values in the same channel.  This is equivalent
# to AveragePool with kernel size equal to the spatial dimension of
# input tensor.
@ConverterRegistry.Register('GlobalAveragePool')
def parse_GlobalAveragePool(graphproto, node, node_name, inputs, outputs):
    caffe_layer = init_node(graphproto, node_name, inputs, outputs)

    stride_h = stride_w = 1
    pad_l = pad_r = pad_t = pad_b = 0

    caffe_layer['functions'] = '{} = L.Pooling({}pad_t={}, pad_b={}, pad_l={}, pad_r={}, '.format(
        caffe_layer['output_name'], caffe_layer['input_names'],
        pad_t, pad_b, pad_l, pad_r)

    caffe_layer['functions'] += 'stride_h={}, stride_w={}, global_pooling=True, '.format(
        stride_h, stride_w)
    caffe_layer['functions'] += 'pool=P.Pooling.AVE)\n'

    return caffe_layer


@ConverterRegistry.Register('Add')
def parse_Add(graphproto, node, node_name, inputs, outputs):
    caffe_layer = init_node(graphproto, node_name, inputs, outputs)

    bias_node = [n for n in graphproto.initializer
                    if n.name == node.input[1]]
    if bias_node != []:
        caffe_layer['bias'] = bias_node
        if len(caffe_layer['bias'][0].float_data) == 1:
            bias = caffe_layer['bias'][0].float_data[0]
            caffe_layer['functions'] = '{} = L.Power({}shift={})\n'.format(
                caffe_layer['output_name'], caffe_layer['input_names'], bias)
        else:
            caffe_layer['bias'] = numpy_helper.to_array(caffe_layer['bias'][0])
            if len(caffe_layer['bias'].shape) == 0:
                caffe_layer['functions'] = '{} = L.Power({}shift={})\n'.format(
                    caffe_layer['output_name'], caffe_layer['input_names'], caffe_layer['bias'])
            else:
                caffe_layer['bias'] = numpy_helper.from_array(caffe_layer['bias'])
                caffe_layer['functions'] = '{} = L.Bias({})\n'.format(
                    caffe_layer['output_name'], caffe_layer['input_names'])
        return caffe_layer

    # if 1st input is the Constant (yolact.onnx)
    add_tensor = [n for n in graphproto.node
                    if n.output[0] == inputs[0]][0]
    if add_tensor.op_type == "Constant":
        addend = numpy_helper.to_array(add_tensor.attribute[0].t)
        caffe_layer['input_names'] = caffe_layer['input_names'].lstrip(', ')
        if len(addend.shape) == 0:
            caffe_layer['functions'] = '{} = L.Power({}shift={})\n'.format(
                caffe_layer['output_name'], caffe_layer['input_names'], addend)
            return caffe_layer
        elif len(addend.shape) == 1:
            caffe_layer['functions'] = '{} = L.Power({}shift={})\n'.format(
                caffe_layer['output_name'], caffe_layer['input_names'], addend[0])
            return caffe_layer
        else:
            caffe_layer['bias'] = addend
            shape = numpy_helper.to_array(caffe_layer['bias']).shape
            if len(shape) == 4:
                caffe_layer['functions'] = '{} = L.Bias({})\n'.format(
                    caffe_layer['output_name'], caffe_layer['input_names'])
            else:
                caffe_layer['functions'] = '{} = L.Bias({}axis=0, num_axes=-1)\n'.format(
                    caffe_layer['output_name'], caffe_layer['input_names'])
            return caffe_layer

    # normal case
    bias = node.input[1]
    caffe_layer['bias'] = [w for w in graphproto.initializer
                           if w.name == bias]
    # second input (initializer) specifies the add data
    if caffe_layer['bias']:
        caffe_layer['bias'] = caffe_layer['bias'][0]
    else:
        bias_node = [n for n in graphproto.node if n.output[0] == bias]
        if bias_node:
            bias_node = bias_node[0]
            if bias_node.op_type == 'Unsqueeze':
                bias = bias_node.input[0]
                caffe_layer['bias'] = [w for w in graphproto.initializer
                                       if w.name == bias][0]
                caffe_layer['input_names'] = caffe_layer['input_names'][:caffe_layer['input_names'].find(",") + 2]
                caffe_layer['functions'] = '{} = L.Bias({})\n'.format(
                    caffe_layer['output_name'], caffe_layer['input_names'])
            else:
                # second input (constant) specifies the add data
                if bias_node.op_type == "Constant":
                    caffe_layer['bias'] = numpy_helper.to_array(bias_node.attribute[0].t)
                    if len(caffe_layer['bias'].shape) == 0:
                        caffe_layer['functions'] = '{} = L.Power({}shift={})\n'.format(
                            caffe_layer['output_name'], caffe_layer['input_names'], caffe_layer['bias'])
                    else:
                        caffe_layer['bias'] = bias_node.attribute[0].t
                        shape = numpy_helper.to_array(caffe_layer['bias']).shape
                        if len(shape) == 4:
                            # for fast rcnn, input shape != Add tensor shape, broadcasting needed
                            # TODO: use Add layer instead
                            caffe_layer['functions'] = '{} = L.Bias({})\n'.format(
                                caffe_layer['output_name'], caffe_layer['input_names'])
                        else:
                            caffe_layer['functions'] = '{} = L.Bias({}axis=0, num_axes=-1)\n'.format(
                                caffe_layer['output_name'], caffe_layer['input_names'])
                else:
                    caffe_layer['functions'] = '{} = L.Add({})\n'.format(
                        caffe_layer['output_name'], caffe_layer['input_names'])
        else:
            caffe_layer['functions'] = '{} = L.Add({})\n'.format(
                caffe_layer['output_name'], caffe_layer['input_names'])

    return caffe_layer

@ConverterRegistry.Register('Reshape')
def parse_Reshape(graphproto, node, node_name, inputs, outputs):
    caffe_layer = init_node(graphproto, node_name, inputs, outputs)

    shape = []
    if len(inputs) == 0:
        # special case for yolov4: this parameter layer can't be merged with single output layer
        if ConverterRegistry.model_name != "yolov4":
            print('skip this layer', outputs)
            caffe_layer['functions'] = '# Skip Reshape\n'
            return caffe_layer
        else:
            param_name = caffe_layer['output_name']
            param_shape = [1, 1, 1, 1]
            caffe_layer['value'] = numpy_helper.from_array(np.array([1]))
            caffe_layer['functions'] = '{} = L.Parameter(shape=dict(dim={}))\n'.format(param_name, param_shape)
            return caffe_layer

    for attr in node.attribute:
        if attr.name == 'shape':
            shape = [int(n) for n in attr.ints]

    # special case hack for m3d: 1st input is Constant,
    # or for tine-yolov3: 1st input is initializer,
    # use parameter layer to store the data
    input_constant = False
    value = []
    if len(inputs) == 1:
        data_tensor = [n for n in graphproto.initializer if n.name == node.input[0]]
        if data_tensor != []:
            data_tensor = data_tensor[0]
            value = data_tensor.float_data
            if value == []:  # raw_data format
                value = numpy_helper.to_array(data_tensor)
            caffe_layer['value'] = numpy_helper.from_array(value)
    else:
        data_node = [n for n in graphproto.node
                     if n.output[0] == inputs[0]]
        if data_node != []:
            data_node = data_node[0]
            if data_node.op_type == "Constant":
                caffe_layer['value'] = data_node.attribute[0].t
                value = numpy_helper.to_array(data_node.attribute[0].t)
    if value != []:
        param_shape = value.shape
        param_name = caffe_layer['output_name'] + ConverterRegistry.CAFFE_NAME_SLASH + 'reshape_input'
        if param_shape == ():
            # for scalar, add one dimension instead of using unsqueeze layer
            param_shape = [1]
        else:
            param_shape = list(param_shape)
        caffe_layer['functions'] = '{} = L.Parameter(shape=dict(dim={}))\n'.format(param_name, param_shape)
        caffe_layer['input_names'] = param_name + ', '
        input_constant = True

    if shape == [] and len(inputs) == 1:
        # second input (initializer) specifies the output shape
        shape_tensor = [n for n in graphproto.initializer
                        if n.name == node.input[1]]
        if shape_tensor != []:
            shape_tensor = shape_tensor[0]
            shape_info = shape_tensor.int64_data
            if shape_info == []:  # raw_data format
                shape_info = shape_tensor.float_data
            shape = [int(n) for n in shape_info]
    if shape == []:
        if len(inputs) > 1:
            # second input (constant) specifies the output shape
            shape_tensor = [n for n in graphproto.node
                            if n.output[0] == inputs[1]]
        else:
            # for tine-yolov3: 1st input is initializer, so the index for shape is changed
            shape_tensor = [n for n in graphproto.node
                            if n.output[0] == inputs[0]]
        if shape_tensor != []:
            shape_tensor = shape_tensor[0]
            if shape_tensor.op_type == "Constant":
                shape = numpy_helper.to_array(shape_tensor.attribute[0].t)
            '''
            else:
                # Tensor value not fixed. May cause dynamic reshape issue for Caffe conversion.
                print("Warning: dynamic reshape occurs for layer {}!".format(node_name))
                # remove the 2nd input
                caffe_layer['input_names'] = caffe_layer['input_names'][:caffe_layer['input_names'].find(",")+2]
                # hack for tiny-yolov3, where Loop layer's output not correctly taken as Reshape input
                if caffe_layer['input_names'] == ', ':
                    caffe_layer['input_names'] = ConverterRegistry.caffe_layer_name(inputs[0]) + ', '

                if ConverterRegistry.dynamic_reshape_dict == None:
                    print("Run onnxruntime to get the dynamic info.")
                    from .get_dynamic_info import run_onnxruntime
                    shape_dict, value_dict = run_onnxruntime(model, ConverterRegistry.user_input_shape)
                    # save as cache to avoid multiple run
                    ConverterRegistry.dynamic_reshape_dict = shape_dict
                    ConverterRegistry.dynamic_value_dict = value_dict
                else:
                    shape_dict = ConverterRegistry.dynamic_reshape_dict

                if node_name in shape_dict.keys():
                    shape = shape_dict[node_name]
                    print("Shape got: {}".format(shape))
                elif outputs[0] in shape_dict.keys():
                    shape = shape_dict[outputs[0]]
                    print("Shape got: {}".format(shape))
                else:
                    print("Shape info not found. Conversion stopped.")
                    quit(1)
            '''

    shape = [int(n) for n in shape]

    if not input_constant:
        caffe_layer['functions'] = "{} = L.Reshape({}reshape_param={{'shape':{{'dim':{}}}}})\n".format(
            caffe_layer['output_name'], caffe_layer['input_names'], shape)
    else:
        caffe_layer['functions'] += "{} = L.Reshape({}reshape_param={{'shape':{{'dim':{}}}}})\n".format(
            caffe_layer['output_name'], caffe_layer['input_names'], shape)

    return caffe_layer

@ConverterRegistry.Register('Softmax')
def parse_Softmax(model, node, node_name, inputs, outputs):
    caffe_layer = init_node(model, node_name, inputs, outputs)
    axis = 1
    for attr in node.attribute:
        if attr.name == 'beta':
            axis = attr.ints[0]
    caffe_layer['functions'] = '{} = L.Softmax({}axis={})\n'.format(
        caffe_layer['output_name'], caffe_layer['input_names'], axis)

    return caffe_layer