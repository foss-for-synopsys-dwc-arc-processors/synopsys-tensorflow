# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Logic to update a TensorFlow model graph with quantization operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import tensorflow as tf
from tensorflow.contrib.quantize.python import common
from tensorflow.contrib.quantize.python import graph_matcher
from tensorflow.contrib.quantize.python import input_to_ops
from tensorflow.contrib.quantize.python import quant_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging

# Quantizable operation types that are supported by the quantization rewrite.
_QUANTIZABLE_TYPES = {'Conv2D', 'MatMul', 'DepthwiseConv2dNative'}

# Activations that are supported by the quantization rewrite.
_ACTIVATION_TYPES = {'Relu', 'Relu6', 'Identity'}

_RELU_TYPES = {'Relu', 'Relu6'}

_QUANTIZATION_OP = {'FakeQuantWithMinMaxVars'}
_VALID_SRC_OP = {'Add', 'Mul'}
_INTERMEDIATE_OP = {'Add', 'Mul'}
_PASS_THROUGH_OP = {'Reshape', 'Identity', 'BatchToSpaceND', 'SpaceToBatchND'}
_VALID_ACTIVATION_OP = {'Relu', 'Relu6'}

# To track Relu/Pool/Concat nodes taking input from FakeQuant node
_BY_PASS_OP = {'Relu', 'Relu6', 'MaxPool', 'ConcatV2', 'AvgPool'}
# Holds the possible input node types for the concat layer.
_INPUT_OP = {'Conv2D', 'FakeQuantWithMinMaxVars', 'Relu', 'Relu6', 'MaxPool', 'ConcatV2', 'AvgPool', 'Identity', 'Reshape', 'MatMul', 'Mul', 'DepthwiseConv2dNative'}
# To get fake quant(act_quant) node or concat node in case of sequential concat layers
_SCALE_PROPAGATE_OP = {'FakeQuantWithMinMaxVars', 'ConcatV2'}

def Quantize(graph,
             is_training,
             weight_bits=8,
             activation_bits=8,
             symmetric=False,
             ev_quant=False,
             ema_decay=0.999,
             quant_delay=None,
             vars_collection=ops.GraphKeys.GLOBAL_VARIABLES,
             scope=None):
  """Updates graph with quantization operations.

  Currently we quantize the following tensors:
  * Conv/MatMul: Quantize the weights if it matches.
  * Activation: Quantize the output if it matches.
  * Bypass/Post-activation Bypass: Quantize both input and output
    if it matches.

  Args:
    graph: Graph to modify.
    is_training: Whether quantizing training graph or eval graph.
    weight_bits: Number of bits to use for quantizing weights.
    activation_bits: Number of bits to use for quantizing activations.
    symmetric: (Optional) If true, use symmetric quantization limits instead of
      training the minimum and maximum of each quantization range separately.
    ema_decay: (Optional) Float, EMA decay parameter.  EMA is used to update
      quantization intervals for quantizing activations (see here about EMA:
      https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average).
    quant_delay: (Optional, default None) Int, count of global steps for which
      to delay quantization.  This helps weights stabilize at the start of
      training.
    vars_collection: (Optional) Collection where to store the variables for
      quantization interval ends.
    scope: The scope to be transformed. If it's not None, only the ops which
      are in this scope will be transformed.
  Raises:
    ValueError: When quantization fails.
  """
  if scope and not scope.endswith('/'):
    scope += '/'
  #Values needed to calculate the output scale in activation node
  w_scale = tf.Variable(initial_value=1.0, trainable=False, name="w_scale")
  ip_scale = tf.Variable(initial_value=0.007843137, trainable=False, name="ip_scale")

  input_to_ops_map = input_to_ops.InputToOps(graph)
  quantized_ops = set()

  # To store the quantizable descendants(Conv, DepthwiseConv, FC) of concat layer
  concat_descendant_list = []
  # To store the context of the quantizable descendants(Conv, DepthwiseConv, FC) of concat layer
  concat_descendant_context_list = []
  # List to hold the concat layers which feds output to quantizable node and takes input from activations (ignoring preprocessing concats)
  valid_concat_list = []
  # Dictionary : key -> concat node name, value -> act quant node name from the inputs of concat node
  concat_dict = dict()
  # Dictionary : key -> quantizable node name, value -> act quant node name from the input of quantizable node
  quantizable_nodes_dict = dict()

  graph_def = graph.as_graph_def()

  # Returns the node with passed node_name
  def fetch_node(node_name):
    for node in graph_def.node:
      if(node.name == node_name):
        return node

  # Finds the quantizable node taking inputs from concat layer
  def find_through_concat_nodes(node_in, quantizable_node_name, quantizable_node_op):
    if(node_in.op is not None and node_in.op == "ConcatV2"):
      if(quantizable_node_name not in quantizable_nodes_dict.keys()):
        concat_descendant_context_list.append(quantizable_node_name.split("/"+quantizable_node_op)[0])
        concat_descendant_list.append(quantizable_node_name)
        quantizable_nodes_dict[quantizable_node_name] = []
        quantizable_nodes_dict[quantizable_node_name].append(node_in.name)
        valid_concat_list.append(node_in.name)
        for iterator in range(node_in.attr["N"].i):
          if("concat" in node_in.input[iterator]):
            valid_concat_list.append(node_in.input[iterator])
        return
    else:
      if(node_in.op in _RELU_TYPES):
        return
      if(node_in.op is not None and node_in.op in _INPUT_OP):
        input_node = fetch_node(node_in.input[0])
        find_through_concat_nodes(input_node, quantizable_node_name, quantizable_node_op)

  count = 0 #To check for the first convolution
  first_node_name = "" #To store the first convolution node name spliting leaf in node name
  for node in graph_def.node:
    if(node.op in _QUANTIZABLE_TYPES and "AuxLogits" not in node.name):
      if(count == 0):
        first_node_name = node.name.split("Conv2D")[0]
        quantizable_nodes_dict[node.name] = [] #Assigning empty list for first convolution with 'Conv2D' leaf in node name
        count = count+1
      else:
        if(node.name.split("Conv2D_Fold")[0] != first_node_name): #Avoid concat node search for first convolution with 'Conv2D_Fold' leaf in node name
          find_through_concat_nodes(node, node.name, node.op)
        else:
          quantizable_nodes_dict[node.name] = [] #Assigning empty list for first convolution with 'Conv2D_Fold' leaf in node name

  for layer_match in _FindLayersToQuantize(graph):
    # Quantize the weights.
    context = _GetContextFromOp(layer_match.layer_op)
    # If `scope` is given, only quantize it if the consumer of weights
    # (the layer op) is in the right scope.
    if layer_match.weight_tensor is not None:
      # Create individual w_scale and ip_scale for descendants of concat
      if(context in concat_descendant_context_list):
        w_scale_concat = tf.Variable(initial_value=0.00, trainable=False, name=context+"/w_scale_concat")
        ip_scale_concat = tf.Variable(initial_value=0.00, trainable=False, name=context+"/ip_scale_concat")
        op_w_scale, op_ip_scale = _InsertQuantOp(
            context,
            'weights_quant',
            layer_match.weight_tensor.op,
            input_to_ops_map.ConsumerOperations(layer_match.weight_tensor.op),
            is_training,
            w_scale_concat,
            ip_scale_concat,
            moving_avg=False,
            ema_decay=ema_decay,
            quant_delay=quant_delay,
            narrow_range=True,
            vars_collection=vars_collection,
            bits=weight_bits,
            tensor_type=0,
            symmetric=symmetric,
            ev_quant=ev_quant,
            consumer_scope=scope)
        scale_w = tf.assign(w_scale_concat, op_w_scale, name=context+"/weights_quant/w_scale")
        scale_ip = tf.assign(ip_scale_concat, op_ip_scale, name=context+"/weights_quant/ip_scale")
      else:
        op_w_scale, op_ip_scale = _InsertQuantOp(
            context,
            'weights_quant',
            layer_match.weight_tensor.op,
            input_to_ops_map.ConsumerOperations(layer_match.weight_tensor.op),
            is_training,
            w_scale,
            ip_scale,
            moving_avg=False,
            ema_decay=ema_decay,
            quant_delay=quant_delay,
            narrow_range=True,
            vars_collection=vars_collection,
            bits=weight_bits,
            tensor_type=0,
            symmetric=symmetric,
            ev_quant=ev_quant,
            consumer_scope=scope)
        scale_w = tf.assign(w_scale, op_w_scale, name=context+"/weights_quant/w_scale")
        scale_ip = tf.assign(ip_scale, op_ip_scale, name=context+"/weights_quant/ip_scale")

    # Quantize the activations.
    if layer_match.activation_op is not None:
      consumer_ops = input_to_ops_map.ConsumerOperations(
          layer_match.activation_op)
      add_context = context
      if layer_match.bypass_op:
        pattern_match_result = re.search(r'^(.*)/([^/]+)', context)
        if pattern_match_result is not None:
          add_context = pattern_match_result.group(1)
        else:
          add_context = ''
      # If `scope` is given, only quantize it if the producer of weights
      # (usually it's the layer op) is in the right scope.
      #Calculated weight scale(w_scale) and activation_scale(ip_scale)
      #from the 'FakeQuantWithMinMaxVars' op
      w_scale, ip_scale = _InsertQuantOp(
          add_context,
          'act_quant',
          layer_match.activation_op,
          consumer_ops,
          is_training,
          scale_w,
          scale_ip,
          moving_avg=True,
          ema_decay=ema_decay,
          quant_delay=quant_delay,
          vars_collection=vars_collection,
          bits=activation_bits,
          tensor_type=1,
          symmetric=symmetric,
          ev_quant=ev_quant,
          init_min=0.0,
          producer_scope=scope)
      quantized_ops.add(layer_match.activation_op)
      weight_scale = tf.Variable(initial_value=0.0, trainable=False,  name=add_context+"/act_quant/w_scale")
      input_scale = tf.Variable(initial_value=0.0, trainable=False, name=add_context+"/act_quant/ip_scale")
      w_scale = tf.assign(weight_scale, w_scale, name=add_context+"/act_quant/w_scale")
      ip_scale = tf.assign(input_scale, ip_scale, name=add_context+"/act_quant/ip_scale")
    # Quantize the inputs and output to the bypass (if it exists). The input to
    # the bypass is the bias add, and the output is the activation.
    if layer_match.bypass_op is not None:
      # If `scope` is given, only quantize it if the both the producer and the
      # consumer are in the right scope.
      _InsertQuantOp(
          context,
          'conv_quant',
          layer_match.bias_add_op,
          input_to_ops_map.ConsumerOperations(layer_match.bias_add_op),
          is_training,
          w_scale,
          ip_scale,
          moving_avg=True,
          ema_decay=ema_decay,
          quant_delay=quant_delay,
          vars_collection=vars_collection,
          bits=activation_bits,
          tensor_type=0,
          symmetric=symmetric,
          ev_quant=ev_quant,
          producer_scope=scope,
          consumer_scope=scope)
      quantized_ops.add(layer_match.bias_add_op)
      # Make sure the op following this isn't an activation. In which case, we
      # shouldn't quantize it, since the activation will be Fused into the
      # Add at inference time.
      consumers = input_to_ops_map.ConsumerOperations(layer_match.bypass_op)
      if any(consumer.type in _ACTIVATION_TYPES for consumer in consumers):
        logging.info('Skipping %s, because its followed by an activation.',
                     layer_match.bypass_op.name)
      else:
        _InsertQuantOp(
            add_context,
            'add_quant',
            layer_match.bypass_op,
            input_to_ops_map.ConsumerOperations(layer_match.bypass_op),
            is_training,
            w_scale,
            ip_scale,
            moving_avg=True,
            ema_decay=ema_decay,
            quant_delay=quant_delay,
            vars_collection=vars_collection,
            bits=activation_bits,
            tensor_type=0,
            symmetric=symmetric,
            ev_quant=ev_quant,
            producer_scope=scope,
            consumer_scope=scope)
        quantized_ops.add(layer_match.bypass_op)

    # Quantize bypass ops that occur after the activation.
    if layer_match.post_activation_bypass_op is not None:
      pattern_match_result = re.search(
          r'^(.*)/([^/]+)', layer_match.post_activation_bypass_op.name)
      if pattern_match_result is not None:
        post_activation_bypass_context = pattern_match_result.group(1)
      else:
        post_activation_bypass_context = ''
      # If `scope` is given, only quantize it if the producer is in the right
      # scope.
      # Make sure the op following this isn't an activation. In which case, we
      # shouldn't quantize it, since the activation will be Fused into the
      # Add at inference time.
      consumers = input_to_ops_map.ConsumerOperations(
          layer_match.post_activation_bypass_op)
      if any(consumer.type in _RELU_TYPES for consumer in consumers):
        logging.info('Skipping %s, because its followed by an activation.',
                     layer_match.post_activation_bypass_op.name)
      else:
        _InsertQuantOp(
            post_activation_bypass_context,
            'post_activation_bypass_quant',
            layer_match.post_activation_bypass_op,
            consumers,
            is_training,
            w_scale,
            ip_scale,
            moving_avg=True,
            ema_decay=ema_decay,
            quant_delay=quant_delay,
            vars_collection=vars_collection,
            bits=activation_bits,
            tensor_type=0,
            symmetric=symmetric,
            ev_quant=ev_quant,
            producer_scope=scope)
        quantized_ops.add(layer_match.post_activation_bypass_op)
  _QuantizeActivationLayers(
      quantized_ops,
      graph,
      is_training,
      symmetric,
      ev_quant,
      activation_bits,
      ema_decay,
      quant_delay,
      vars_collection,
      scope=scope)
  #To Reroute the input scale nodes correctly based on its order
  if(ev_quant):
    graph_def = graph.as_graph_def()
    act_quant = {}
    for node in graph_def.node: #Searching for all the activation nodes and save the name as 'key' in a dictionary
      if(node.op in _QUANTIZATION_OP):
        if(("act_quant/FakeQuantWithMinMaxVars" in node.name) or ("activation_" in node.name)):
          act_quant[node.name] = []
    for node in graph_def.node: #Filling dict 'values' with list of nodes getting inputs from 'key'
      if(node.op in _QUANTIZABLE_TYPES):
        if(node.input[0] in act_quant.keys()):
            act_quant[node.input[0]].append(node.name)
      elif(node.op in _BY_PASS_OP):#Relu/MaxPool taking input from fakequant node
        if(node.input[0] in act_quant.keys()):
          for nxt_node in graph_def.node:#So another loop to figure out the corresponding Fakequant node to Relu, MaxPool (to get ip scale, weight_scale)
            if(nxt_node.op in _QUANTIZABLE_TYPES):
              if(node.name in nxt_node.input[0]):
                act_quant[node.input[0]].append(nxt_node.name)
    for key, value in sorted(act_quant.items()): #Deleting 'keys' which is being passed as an input to only one node
      if(len(value) < 2):
        del act_quant[key]

    # Traverse to get fake quant(act_quant) node or concat node in case of sequential concat layers
    # Maintains the fake quant node of all the concat layer inputs under the corresponding concat node name as key
    def find_scale_propagate_nodes(node_in, dictionary, dict_entry):
      if(node_in.op is not None and node_in.op in _SCALE_PROPAGATE_OP):
        dictionary[dict_entry].append(node_in.name)
        return
      else:
        if(node_in.op is not None and node_in.op in _INPUT_OP):
          node = fetch_node(node_in.input[0])
          find_scale_propagate_nodes(node, dictionary, dict_entry)

    # Updates the fake quant node for concat inputs
    def fill_concat_dict():
      add_list = []
      remove_list = []
      # Append Fake quant node from the inputs of concat layer
      for concat_node in graph_def.node:
        if(concat_node.op == "ConcatV2" and concat_node.name in valid_concat_list):
          if ("N" in concat_node.attr):
            concat_dict[concat_node.name] = []
            # Iterates until number of inputs to concat node
            for iterator in range(concat_node.attr["N"].i):
              # Fetch the input node for the concat layer
              input_node = fetch_node(concat_node.input[iterator])
              find_scale_propagate_nodes(input_node, concat_dict, concat_node.name)
      # Append the concat list in case of sequential concat layers (i.e concat -> concat)
      for concat_key in concat_dict.keys():
        # Iterates for number of inputs to concat layer
        for iterator in range(len(concat_dict[concat_key])):
          # Check if the input of concat layer is another concat layer
          if(concat_dict[concat_key][iterator] in concat_dict.keys()):
            branch_concat = concat_dict[concat_key][iterator]
            # Add the fake quant node of input concat layer to add list
            for iterator1 in range(len(concat_dict[branch_concat])):
              add_list.append(concat_dict[branch_concat][iterator1])
            # Add the input concat name to the remove list, in order to remove it from the final concat node
            remove_list.append(branch_concat)
        for val in add_list:
          concat_dict[concat_key].append(val)
        for val in remove_list:
          concat_dict[concat_key].remove(val)
        add_list.clear()
        remove_list.clear()
      return concat_dict

    def max_of_concat_inputs(concat_name, concat_dict):
      # Tensors to hold max value and scale value from the inputs of concat layer
      max_tensor = tf.Variable(initial_value=0.0, name="MaxTensor")
      scale_tensor = tf.Variable(initial_value=0.0, name="ScaleTensor")
      # Iterates for inputs of concat layer and updates the maxtensor
      for iterator in range(len(concat_dict[concat_name])):
        for in_node in graph_def.node:
          if(in_node.name == concat_dict[concat_name][iterator]):
            input_tensor = graph.get_tensor_by_name(in_node.input[2]+":0")
            if(not iterator):
              max_tensor = tf.assign(max_tensor, input_tensor)
            else:
              max_tensor = tf.cond(tf.greater(input_tensor, max_tensor), lambda: input_tensor, lambda: max_tensor)
            break
      # Iterates for inputs of concat layer and updates the scaletensor with the scale of maxtensor
      for iterator in range(len(concat_dict[concat_name])):
        for in_node in graph_def.node:
          if(in_node.name == concat_dict[concat_name][iterator]):
            input_tensor = graph.get_tensor_by_name(in_node.input[2]+":0")
            scale_tensor_1 = graph.get_tensor_by_name(concat_dict[concat_name][iterator].split("/FakeQuantWithMinMaxVars")[0] + "/ip_scale_1:0")
            if(not iterator):
              scale_tensor = tf.assign(scale_tensor, scale_tensor_1)
            scale_tensor = tf.cond(tf.equal(input_tensor, max_tensor), lambda: scale_tensor_1, lambda: scale_tensor)
            break
      return scale_tensor

    # Fills the fake quant node for corresponding quantizable nodes(Conv, Depthwiseconv, FC)
    for node in graph_def.node:
      if(node.op in _QUANTIZABLE_TYPES and node.name not in quantizable_nodes_dict.keys() and "AuxLogits" not in node.name):
        quantizable_nodes_dict[node.name] = []
        find_scale_propagate_nodes(node, quantizable_nodes_dict, node.name)

    # Fills concat-dict with fake quant node corresponding to the inputs of concat
    concat_dict = fill_concat_dict()
    # Reroutes the ip_scale and w_scale from fakequant(act quant) node stored in quantizable_nodes_dict
    for key_val in sorted(quantizable_nodes_dict.keys()):
      quantizable_node = fetch_node(key_val)
      dict_value = quantizable_nodes_dict[key_val]
      for in_node in graph_def.node:
        # Check for weight_quant corresponding to the quantizable node
        if(in_node.op in _QUANTIZATION_OP and in_node.name == quantizable_node.input[1]):
          dst_tensor = graph.get_tensor_by_name(in_node.input[4]+":0")
          # Assigns scale tensor from the max of concat inputs
          if(len(dict_value) > 0):
            if(key_val in concat_descendant_list and dict_value[0] in concat_dict.keys()):
              scale_tensor = max_of_concat_inputs(dict_value[0], concat_dict)
              # Reroutes ip_scale in weight quant from the fetched scale tensor
              # w_scale wont be updated for concat descendants
              common.RerouteTensor(scale_tensor, dst_tensor)
            # Reroutes ip_scale and w_scale for branching points from conv layer
            else:
              sub_name = dict_value[0].split("/FakeQuantWithMinMaxVars")[0]
              if((sub_name not in in_node.input[3] and sub_name not in in_node.input[4]) and (in_node.input[3] != "w_scale/read" and in_node.input[4] != "ip_scale/read")):
                # To reroute ip_scales
                src_tensor = graph.get_tensor_by_name(sub_name+"/ip_scale_1:0")
                common.RerouteTensor(src_tensor, dst_tensor)
                if(sub_name in str(act_quant.keys())): #To create individual copy for w_scale (for branching points)
                  w_scale_new = tf.Variable(initial_value=0.0, trainable=False,  name=in_node.name.split("/weights_quant/FakeQuantWithMinMaxVars")[0]+"/w_scale_2")
                  common.RerouteTensor(
                          graph.get_tensor_by_name(in_node.name.split("/weights_quant/FakeQuantWithMinMaxVars")[0]+"/w_scale_2:0"), graph.get_tensor_by_name(in_node.input[3]+":0"))
                else: #To reroute w_scale without creating a copy
                  common.RerouteTensor(
                          graph.get_tensor_by_name(sub_name+"/w_scale_1:0"), graph.get_tensor_by_name(in_node.input[3]+":0"))

def _QuantizeActivationLayers(quantized_ops,
                              graph,
                              is_training,
                              symmetric,
                              ev_quant,
                              activation_bits=8,
                              ema_decay=0.999,
                              quant_delay=None,
                              vars_collection=ops.GraphKeys.GLOBAL_VARIABLES,
                              scope=None):
  """Quantize intermediate activation tensors after addition and multiplication.
  Args:
    quantized_ops: Set of previously quantized activation ops.
    graph: Graph to modify.
    is_training: Whether quantizing training graph or eval graph.
    activation_bits: Number of bits to use for quantizing activations.
    ema_decay: (Optional) Float, EMA decay parameter.  EMA is used to update
      quantization intervals for quantizing activations (see here about EMA:
      https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average).
    quant_delay: (Optional, default None) Int, count of global steps for which
      to delay quantization.  This helps weights stabilize at the start of
      training.
    vars_collection: (Optional) Collection where to store the variables for
      quantization interval ends.
    scope: The scope to be transformed. If it's not None, only the ops which are
      in this scope will be transformed.
  Raises:
    ValueError: When quantization fails.
  """
  w_scale_act = tf.Variable(initial_value=0.0, trainable=False, name="w_scale_act")
  ip_scale_act = tf.Variable(initial_value=0.0, trainable=False, name="ip_scale_act")
  input_to_ops_map = input_to_ops.InputToOps(graph)
  for op in (op for op in graph.get_operations()):
    if _CheckIfQuantizableOp(op, quantized_ops):
      logging.info('Inserting fake quant op activation_%s_quant after %s',
                   op.type, op.name)
      consumers = input_to_ops_map.ConsumerOperations(op)
      op_w_scale_act, op_ip_scale_act = _InsertQuantOp(
          op.name,
          'activation_' + op.type + '_quant',
          op,
          consumers,
          is_training,
          w_scale_act,
          ip_scale_act,
          moving_avg=True,
          tensor_type=2,
          symmetric=symmetric,
          ev_quant=ev_quant,
          ema_decay=ema_decay,
          quant_delay=quant_delay,
          vars_collection=vars_collection,
          bits=activation_bits,
          producer_scope=scope)
      weight_scale = tf.Variable(initial_value=0.0, trainable=False,  name=op.name+'/activation_' + op.type + '_quant/w_scale')
      input_scale = tf.Variable(initial_value=0.0, trainable=False, name=op.name+'/activation_' + op.type + '_quant/ip_scale')
      w_scale_act_assign = tf.assign(weight_scale, op_w_scale_act, name=op.name+'/activation_' + op.type + '_quant/w_scale')
      ip_scale_act_assign = tf.assign(input_scale, op_ip_scale_act, name=op.name+'/activation_' + op.type + '_quant/ip_scale')

def _CheckIfQuantizableOp(src_op, quantized_ops):
  """Check if the output of an op should be quantized.
  Args:
    src_op: op to be checked
    quantized_ops: Set of previously quantized activation ops.
  Returns:
    Boolean specifying if output should be quantized or not.
  """
  src_op_name = set([src_op.type])
  if src_op in quantized_ops:
    return False
  if not src_op_name.intersection(_VALID_SRC_OP):
    return False

  # If src op is an add or a mul and the output is immediately
  # followed by an activation skip
  if len(src_op.outputs) == 1 and len(src_op.outputs[0].consumers()) == 1:
    op_consumers = src_op.outputs[0].consumers()
    if set([op_consumers[0].type]).intersection(_VALID_ACTIVATION_OP):
      logging.info('Skipping quant after %s', src_op.name)
      return False
  # Is an Add or a Mul
  input_ops = src_op.inputs

  for op in input_ops:
    curr_op = op.op
    curr_op_type = set([curr_op.type])
    while curr_op_type.intersection(_PASS_THROUGH_OP):
      # Walk back through pass through ops
      curr_op = curr_op.inputs[0].op
      curr_op_type = set([curr_op.type])
      # Now at a valid or quantizable op, need to check if
      # atleast one of the inputs to a valid op is connected
      # to a quantizable op via pass through ops

    if (curr_op_type.intersection(_QUANTIZATION_OP) or
        curr_op.name.find('delayed_quant/Merge') > 0):
      return True

    if curr_op_type.intersection(_INTERMEDIATE_OP):
      # Check if atleast one input to intermediate_op are quantizable
      for input_op in curr_op.inputs:
        if _CheckIfQuantizableOp(input_op.op, quantized_ops):
          return True
  return False


def _FindLayersToQuantize(graph):
  """Matches layers in graph to quantize.
  The following patterns get matched. Nodes surrounded by [] will be
  optionally matched:
          weight|folded_weight
                /
         conv|fc
            |
      [batch_to_space_nd]
            |
    [post_conv_correction]
            |
     [biasadd|folded_bias]
            |
         [bypass]
            |
        activation
            |
   [post_activation_bypass]
  Match replacements:
    If weight|folded_weight is found, FakeQuant is added afterwards.
    If bypass is found, FakeQuant is added before and after.
    If activation is found, FakeQuant is added afterwards.
    If post_activation_bypass is found, FakeQuant is added afterwards.

  Args:
    graph: Graph to perform match on.

  Returns:
    list of _LayerMatches.
  """
  input_pattern = graph_matcher.OpTypePattern('*')
  weight_var_pattern = graph_matcher.OpTypePattern('Variable|VariableV2')
  weight_partition_identity_pattern = graph_matcher.OpTypePattern(
      'Identity', inputs=[weight_var_pattern])
  weight_partition_concat_pattern = graph_matcher.OpTypePattern(
      'ConcatV2', inputs=[weight_partition_identity_pattern, '*', '*'])
  weight_identity_pattern = graph_matcher.OpTypePattern(
      'Identity',
      inputs=[
          graph_matcher.OneofPattern([
              weight_partition_identity_pattern,
              weight_partition_concat_pattern,
              weight_var_pattern,
          ])
      ])
  weight_resource_var_pattern = graph_matcher.OpTypePattern('ReadVariableOp')
  folded_weight_pattern = graph_matcher.OpTypePattern('Mul')

  # The weights inputs to the layer operation can either be from the Variable or
  # the folded weight (Mul).
  layer_pattern = graph_matcher.OpTypePattern(
      '|'.join(_QUANTIZABLE_TYPES),
      inputs=[
          input_pattern,
          graph_matcher.OneofPattern([
              weight_identity_pattern, weight_resource_var_pattern,
              folded_weight_pattern
          ])
      ],
      ordered_inputs=False)

  # For atrous convolutions a BatchToSpaceND will occur after the depthwise
  # convolution.
  batch_to_space_pattern = graph_matcher.OpTypePattern(
      'BatchToSpaceND',
      inputs=[
          layer_pattern,
          graph_matcher.OpTypePattern('*'),
          graph_matcher.OpTypePattern('*')
      ])

  layer_output_pattern = graph_matcher.OneofPattern(
      [batch_to_space_pattern, layer_pattern])

  # For separable convolutions, we are looking for a conv, followed by a conv
  # with no activations between the two.
  sep_conv_pattern = graph_matcher.OpTypePattern(
      '|'.join(_QUANTIZABLE_TYPES),
      inputs=[
          graph_matcher.OneofPattern([layer_output_pattern]),
          graph_matcher.OpTypePattern('*')
      ],
      ordered_inputs=False)
  folded_bias_mul_pattern = graph_matcher.OpTypePattern(
      'Mul',
      inputs=[graph_matcher.OpTypePattern('*'), layer_output_pattern],
      ordered_inputs=False)
  post_layer_op_correction_pattern = graph_matcher.OpTypePattern(
      'Add',
      inputs=[folded_bias_mul_pattern,
              graph_matcher.OpTypePattern('*')],
      ordered_inputs=False)
  folded_bias_add_pattern = graph_matcher.OpTypePattern(
      'Add',
      inputs=[
          post_layer_op_correction_pattern,
          graph_matcher.OpTypePattern('*')
      ],
      ordered_inputs=False)

  # batch_norms with forced updates have an Identity operation at the end.
  # TODO(suharshs): Find a way to easily skip extra Identity operations. The
  # current issue is that doing so can often match patterns across many layers
  # incorrectly.
  batch_norm_identity = graph_matcher.OpTypePattern(
      'Identity', inputs=[folded_bias_add_pattern])

  bias_add_pattern = graph_matcher.OpTypePattern(
      'Add|BiasAdd', inputs=[layer_output_pattern, '*'], ordered_inputs=False)

  # The bias can come from the bias add or the folded bias add.
  bypass_pattern = graph_matcher.OpTypePattern(
      'Add',
      inputs=[
          graph_matcher.OneofPattern(
              [bias_add_pattern, folded_bias_add_pattern, batch_norm_identity]),
          '*'
      ],
      ordered_inputs=False)

  # The input to the activation can come from bias add, fold bias add, the
  # bypasses.
  # TODO(suharshs): We should ideally skip Identity operations instead of
  # treating them as activations.
  activation_pattern = graph_matcher.OpTypePattern(
      '|'.join(_ACTIVATION_TYPES) + '|Identity',
      inputs=[
          graph_matcher.OneofPattern([
              bias_add_pattern,
              folded_bias_add_pattern,
              batch_norm_identity,
              bypass_pattern,
              layer_pattern,
          ])
      ])

  post_activation_bypass_pattern = graph_matcher.OpTypePattern(
      'Add', inputs=['*', activation_pattern], ordered_inputs=False)

  # The order of the following matching blocks is very important. Since matches
  # aren't guaranteed to be disjoint, we structure matches from largest to
  # smallest to guarantee that the largest match always wins. Additionally, we
  # ensure that we don't match layers multiple times.

  layer_matches = []
  # We use matched_layer_set to ensure that layers aren't matched multiple
  # times.
  matched_layer_set = set()

  # First, we match layers that have a post activation bypass. We do this first
  # to ensure we don't match only the first part of this layer, missing the
  # post activation bypass node.
  post_activation_bypass_layer_matcher = graph_matcher.GraphMatcher(
      post_activation_bypass_pattern)
  for match_result in post_activation_bypass_layer_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_tensor = match_result.get_tensor(weight_identity_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(weight_resource_var_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(folded_weight_pattern)
    activation_op = match_result.get_op(activation_pattern)
    bias_add_op = match_result.get_op(bias_add_pattern)
    if bias_add_op is None:
      bias_add_op = match_result.get_op(folded_bias_add_pattern)
    bypass_op = match_result.get_op(bypass_pattern)
    post_activation_bypass_op = match_result.get_op(
        post_activation_bypass_pattern)
    if layer_op not in matched_layer_set:
      matched_layer_set.add(layer_op)
      layer_matches.append(
          _LayerMatch(layer_op, weight_tensor, activation_op, bypass_op,
                      post_activation_bypass_op, bias_add_op))

  # Now, we match the basic layer ending at an activation. We may get duplicate
  # matches from above, but we don't add them to layer_matches.
  layer_matcher = graph_matcher.GraphMatcher(activation_pattern)
  for match_result in layer_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_tensor = match_result.get_tensor(weight_identity_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(weight_resource_var_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(folded_weight_pattern)
    activation_op = match_result.get_op(activation_pattern)
    bias_add_op = match_result.get_op(bias_add_pattern)
    if bias_add_op is None:
      bias_add_op = match_result.get_op(folded_bias_add_pattern)
    bypass_op = match_result.get_op(bypass_pattern)
    if layer_op not in matched_layer_set:
      if not _IsSkipLayer(activation_op):
        matched_layer_set.add(layer_op)
        layer_matches.append(
            _LayerMatch(layer_op, weight_tensor, activation_op, bypass_op, None,
                        bias_add_op))

  # Match the final layer, where there may not be an activation and instead
  # the output of the final BiasAdd must be quantized. So we treat the BiasAdd
  # as the 'activation_op' in the _LayerMatch, to ensure that it's output is
  # quantized.
  final_layer_matcher = graph_matcher.GraphMatcher(
      graph_matcher.OneofPattern([bias_add_pattern, folded_bias_add_pattern]))
  for match_result in final_layer_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_tensor = match_result.get_tensor(weight_identity_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(weight_resource_var_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(folded_weight_pattern)
    activation_op = match_result.get_op(bias_add_pattern)
    if activation_op is None:
      activation_op = match_result.get_op(folded_bias_add_pattern)
    if layer_op not in matched_layer_set:
      matched_layer_set.add(layer_op)
      layer_matches.append(
          _LayerMatch(layer_op, weight_tensor, activation_op, None, None, None))

  # Look for separable convolutions here
  sep_conv_matcher = graph_matcher.GraphMatcher(sep_conv_pattern)
  for match_result in sep_conv_matcher.match_graph(graph):
    layer_op = match_result.get_op(layer_pattern)
    weight_tensor = match_result.get_tensor(weight_identity_pattern)
    if weight_tensor is None:
      weight_tensor = match_result.get_tensor(weight_resource_var_pattern)
    activation_op = match_result.get_op(layer_pattern)
    if layer_op not in matched_layer_set:
      matched_layer_set.add(layer_op)
      layer_matches.append(
          _LayerMatch(layer_op, weight_tensor, activation_op, None, None, None))
  return layer_matches

def _IsSkipLayer(activation_op):
  """Skip quantizing conv->identity->Batch norm layers.
  Args:
    activation_op: Activation op detected by layer matching pattern
  Returns:
    skip_layer: boolean, true when conv->identity->batch norm is detected.
  """

  # Exclude quantization of conv->identity->BN,
  # After folding, this part corresponds to estimation of mean and variance
  # and should not be quantized.
  skip_layer = False
  if activation_op.type == 'Identity' and len(activation_op.outputs) == 1:
    if len(activation_op.outputs[0].consumers()) == 1:
      consumer = activation_op.outputs[0].consumers()[0]
      if consumer.type in ['FusedBatchNorm', 'FusedBatchNormV3']:
        skip_layer = True
        logging.info(
            'Skipping quantizing %s, because it is the output of a conv/fc '
            'followed by a identity, feeding a fused batch norm.',
            activation_op.name)
  return skip_layer

class _LayerMatch(object):
  """Contains all information related to a matched Layer."""

  def __init__(self, layer_op, weight_tensor, activation_op, bypass_op,
               post_activation_bypass_op, bias_add_op):
    self._layer_op = layer_op
    self._weight_tensor = weight_tensor
    self._activation_op = activation_op
    self._bypass_op = bypass_op
    self._post_activation_bypass_op = post_activation_bypass_op
    self._bias_add_op = bias_add_op

  @property
  def layer_op(self):
    return self._layer_op

  @property
  def weight_tensor(self):
    return self._weight_tensor

  @property
  def activation_op(self):
    return self._activation_op

  @property
  def bypass_op(self):
    return self._bypass_op

  @property
  def post_activation_bypass_op(self):
    return self._post_activation_bypass_op

  @property
  def bias_add_op(self):
    return self._bias_add_op

def _FollowedByFakeQuant(tensor):
  """Returns True if the tensor is followed by a FakeQuant."""
  fake_quant_ops = set([
      'FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxArgs',
      'FakeQuantWithMinMaxVarsPerChannel'
  ])
  pass_through_ops = set(['Reshape', 'Identity'])
  consumers = tensor.consumers()
  while consumers:
    c = consumers.pop()
    if c.type in fake_quant_ops:
      return True
    elif c.type in pass_through_ops:
      for output in c.outputs:
        consumers.extend(output.consumers())
  return False

def _InsertQuantOp(context,
                   name,
                   producer,
                   consumers,
                   is_training,
                   w_scale=1.0,
                   ip_scale=0.007843137,
                   moving_avg=True,
                   init_min=-6.0,
                   init_max=6.0,
                   bits=8,
                   tensor_type=0,
                   symmetric=False,
                   ev_quant=False,
                   ema_decay=0.999,
                   quant_delay=None,
                   vars_collection=ops.GraphKeys.GLOBAL_VARIABLES,
                   narrow_range=False,
                   producer_scope=None,
                   consumer_scope=None):
  """Inserts a quant op between a producer op and (multiple) consumer ops.
  Args:
    context: Context where producer and consumer operations are nested.
    name: Name for the new quantization op within the context.
    producer: Producer operation of the pairs where quantization will be
      inserted.
    consumers: Consumer operations of the pairs.
    is_training: Whether quantizing training graph or eval graph.
    moving_avg: Specifies whether to use exponential moving average or just
      the last value seen.
    init_min: Starting minimum value for the new quantization op.
    init_max: Starting maximum value for the new quantization op.
    bits: Number of bits to use for quantization, must be between 2 and 8.
    symmetric: (Optional) If true, use symmetric quantization limits instead of
      training the minimum and maximum of each quantization range separately.
    ema_decay: (Optional) Float, EMA decay parameter.  EMA is used to update
      quantization intervals for quantizing activations (see here about EMA:
      https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average).
    quant_delay: (Optional, default None) Int, count of global steps for which
      to delay quantization.  This helps weights stabilize at the start of
      training.
    vars_collection: (Optional) Collection where to store the variables for
      quantization interval ends.
    narrow_range: Whether to use the narrow quantization range
      [1; 2^bits - 1] or wide range [0; 2^bits - 1].
    producer_scope: The restriction of producer scope. If not None, the new op
      will be inserted only when the producer is in this scope.
    consumer_scope: The restriction of consumer scope. If not None, the new op
      will be inserted only when all the consumers are in this scope.
  Raises:
    ValueError: When producer operation is not directly connected to the
      consumer operation.
  """
  if producer_scope and not producer.name.startswith(producer_scope):
    logging.info(
        '_InsertQuantOp ignores context="%s" name="%s" '
        'because producer "%s" is not in scope "%s"',
        context, name, producer.name, producer_scope)
    return

  if consumer_scope:
    consumers_in_scope = []
    for consumer in consumers:
      if consumer.name.startswith(consumer_scope):
        consumers_in_scope.append(consumer)
      else:
        logging.info(
            '_InsertQuantOp context="%s" name="%s" ignores '
            'consumer "%s" because it is not in scope "%s"',
            context, name, consumer.name, consumer_scope)
        return
    consumers = consumers_in_scope

  name_prefix = _AddContextToName(context, name)
  # This is needed on TPU where name_scope == 'TPUReplicate/loop', and
  # name_prefix starts with 'TPUReplicate/loop/'; without dropping it
  # variables are created as TPUReplicate/loop/TPUReplicate/loop/..., which
  # breaks things later.
  name_scope = ops.get_name_scope()
  if name_scope:
    name_prefix = common.DropStringPrefix(name_prefix, name_scope + '/')

  inputs = producer.outputs[0]
  # Prevent ops from being quantized multiple times. Bypass ops can sometimes
  # overlap between multiple matches, so we need to ensure that we don't
  # add duplicate FakeQuant operations.
  if _FollowedByFakeQuant(inputs):
    return
  if moving_avg:
    quant = (
        quant_ops.MovingAvgQuantize(
            inputs,
            w_scale,
            ip_scale,
            init_min=init_min,
            init_max=init_max,
            ema_decay=ema_decay,
            is_training=is_training,
            num_bits=bits,
            tensor_type=tensor_type,
            symmetric=symmetric,
            ev_quant=ev_quant,
            narrow_range=narrow_range,
            vars_collection=vars_collection,
            name_prefix=name_prefix))
  else:
     quant = (
        quant_ops.LastValueQuantize(
            inputs,
            w_scale,
            ip_scale,
            init_min=init_min,
            init_max=init_max,
            is_training=is_training,
            num_bits=bits,
            tensor_type=tensor_type,
            symmetric=symmetric,
            ev_quant=ev_quant,
            narrow_range=narrow_range,
            vars_collection=vars_collection,
            name_prefix=name_prefix))

  quant_0 = quant[0]
  quant_1 = quant[1]
  quant_2 = quant[2]
  if quant_delay and quant_delay > 0:
    activate_quant = math_ops.greater_equal(
        common.CreateOrGetQuantizationStep(),
        quant_delay,
        name=name_prefix + '/activate_quant')
    quant_0 = control_flow_ops.cond(
        activate_quant,
        lambda: quant_0,
        lambda: inputs,
        name=name_prefix + '/delayed_quant')

  if consumers:
    tensors_modified_count = common.RerouteTensor(
        quant_0, inputs, can_modify=consumers)
    # Some operations can have multiple output tensors going to the same
    # consumer. Since consumers is a set, we need to ensure that
    # tensors_modified_count is greater than or equal to the length of the set
    # of consumers.
    if tensors_modified_count < len(consumers):
      raise ValueError('No inputs quantized for ops: [%s]' % ', '.join(
          [consumer.name for consumer in consumers]))
  return quant_1, quant_2

def _GetContextFromOp(op):
  """Gets the root context name from the op name."""
  context_re = re.search(r'^(.*)/([^/]+)', op.name)
  if context_re:
    return context_re.group(1)
  return ''

def _AddContextToName(context, name):
  """Adds the context to the name if it exists."""
  if not context:
    return name
  return context + '/' + name
