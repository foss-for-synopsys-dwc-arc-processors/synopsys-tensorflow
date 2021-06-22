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

import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
from google.protobuf import text_format
import os
import re
import platform
import networkx as nx
from .parseMWNN import ConverterRegistry
import sys
sys.path.append("../../")
from mwnn_protobuf.python_wrapper import MWNN_pb2


def input_in_skip_list(graphproto, node_name, skip_list, anonym):
    node = [n for n in graphproto.node
            if n.output[0] == node_name or n.name == node_name][0]
    for input_name in node.input:
        if input_name in skip_list:
            return True, input_name
        elif input_name in anonym.keys():
            if anonym[input_name] in skip_list:
                return True, input_name
    return False


def convert(mwnn_file_path, output_folder, last_nodes=[], input_shape=[], first_nodes=[]):

    if mwnn_file_path is None or not os.path.exists(mwnn_file_path):
        print('*' * 80)
        print("Model file doesn't exist:", mwnn_file_path)
        print('*' * 80)
        quit(1)
    print(mwnn_file_path)
    file1 = open(mwnn_file_path, "rb") #read mWNN graph proto
    graphproto = MWNN_pb2.MWNNGraphProto()
    graphproto.ParseFromString(file1.read())

    op_types = set()
    for n in graphproto.node:
        if n.op_type not in op_types:
            op_types.add(n.op_type)
    print(op_types, "\n")

    ops_types_supported = set(ConverterRegistry.registry_.keys())
    op_types_left = op_types - ops_types_supported
    print("Op types that are possibly not supported yet:")
    print(op_types_left, "\n")

    # if the path is on windows platform
    if platform.system() == "Windows":
        mwnn_file = mwnn_file_path.split('\\')[-1].rstrip('.bin')
    else:
        mwnn_file = mwnn_file_path.split('/')[-1].rstrip('.bin')

    initializers = [d.name for d in graphproto.initializer]

    # G = nx.DiGraph()
    mwnn_converter = ConverterRegistry(mwnn_file, graphproto, input_shape, first_nodes)
    caffe_network = mwnn_converter.caffe_network

    caffe_layers = ""

    # last_nodes (subgraph cut) processing
    skip_nodes = []
    print("first_nodes: ", first_nodes)
    print("last_nodes: ", last_nodes)
    '''
    if last_nodes != [] or first_nodes != []:
        all_node_name = []
        for node in graphproto.node:
            node_name = node.name if node.name else node.output[0]
            all_node_name.append(node_name)

        for l in last_nodes:
            if l not in all_node_name:
                print("Warning: {} is not a valid last node name assignment for the EV model conversion. "
                      "Please recheck your input!\n"
                      "Note: Please use the Node name instead of Node outputs' name if they differ.\n".format(l))
        for f in first_nodes:
            if f not in all_node_name:
                print("Warning: {} is not a valid first node name assignment for the EV model conversion. "
                      "Please recheck your input!\n"
                      "Note: Please use the Node name instead of Node outputs' name if they differ.\n".format(f))
        ### DiGraph object (G) used ony to check the validity of first nodes and last nodes. Those are empty as it's not passed
        G = nx.DiGraph()
        for node in graphproto.node:
            node_name = node.name if node.name else node.output[0]
            G.add_node(node_name, name=node_name)
            for i in range(len(node.input)):
                G.add_edge(node.input[i], node_name)
                if node_name != node.output[0]:
                    G.add_edge(node_name, node.output[0])
                    if len(node.output) > 1:
                        for i in range(len(node.output)):
                            G.add_edge(node_name, node.output[i])

        if last_nodes != []:
            for n in G.nodes:
                if all([not nx.has_path(G, n, l) for l in last_nodes]):
                    skip_nodes.append(n)

        if first_nodes != []:
            for n in G.nodes:
                # remove all the previous nodes; but still include the assigned first nodes
                if any([nx.has_path(G, n, f) for f in first_nodes]) and n not in first_nodes:
                    skip_nodes.append(n)

        print("Skip parsing:", skip_nodes, "\n")

    # TODO: customized assignment
    ignore_branches = []
    if mwnn_file == "yolov4_1_3_608_608":
        ignore_branches = ["Shape_399", "Gather_400", "Sub_401", "ConstantOfShape_402", "Concat_403", "Reshape_405",
                           "Slice_410", "Transpose_411", "Reshape_413", "Cast_414",
                           "Shape_419", "Gather_420", "Sub_421", "ConstantOfShape_422", "Concat_423", "Reshape_425",
                           "Slice_430", "Transpose_431", "Reshape_433", "Cast_434",
                           "Shape_439", "Gather_440", "Sub_441", "ConstantOfShape_442", "Concat_443", "Reshape_445",
                           "Slice_450", "Transpose_451", "Reshape_453", "Cast_454",

                           "Constant_472", "Shape_473", "ConstantOfShape_474", "Mul_476", "Equal_478", "Where_479",
                           "Constant_507", "Shape_508", "ConstantOfShape_509", "Mul_511", "Equal_513", "Where_514"]
    '''
    ignore_branches = []
    caffe_last_nodes = []
    caffe_first_nodes = []
    implicit_last_nodes = []

    # list the pre last node types for handling
    Skip_last_node_types = ['Identity', 'Cast', 'Dropout', 'AliasWithName', 'Constant']
    # set default last nodes for caffe optimization if none is given
    if last_nodes == []:
        singularity = 0
        mwnn_last_nodes = [n.name for n in graphproto.output]
        mwnn_first_nodes = [n.name for n in graphproto.input]
        for node in graphproto.node:
            node_name = node.name if node.name else node.output[0]
            if node.output[0] in mwnn_last_nodes:
                if node.op_type not in Skip_last_node_types:
                    implicit_last_nodes.append(node_name)
                else:
                    if len(node.input) == 0:
                        print("The node {} is isolated. Ignore it in last node list.\n".format(node_name))
                        singularity += 1
                        continue
                    pre_node_name = node.input[0]
                    if pre_node_name in mwnn_first_nodes:
                        print("The node {} is isolated. Ignore it in last node list.\n".format(node_name))
                        singularity += 1
                        continue
                    for pre_node in graphproto.node:
                        if pre_node.name == pre_node_name or pre_node.output[0] == pre_node_name:
                            pre_node_name = pre_node.name if pre_node.name else pre_node.output[0]
                            if pre_node.op_type not in Skip_last_node_types:
                                implicit_last_nodes.append(pre_node_name)
                            else:
                                # maximum search back 2 nodes deep
                                if len(pre_node.input) == 0:
                                    print("The node {} is isolated. Ignore it in last node list.\n".format(
                                        pre_node_name))
                                    singularity += 1
                                    continue
                                pre_pre_node_name = pre_node.input[0]
                                if pre_pre_node_name in mwnn_first_nodes:
                                    print("The node {} is isolated. Ignore it in last node list.\n".format(
                                        pre_node_name))
                                    singularity += 1
                                    continue
                                for pre_pre_node in graphproto.node:
                                    if pre_pre_node.name == pre_pre_node_name or \
                                            pre_pre_node.output[0] == pre_pre_node_name:
                                        pre_pre_node_name = pre_pre_node.name if pre_pre_node.name else \
                                            pre_pre_node.output[0]
                                        if pre_pre_node.op_type not in Skip_last_node_types:
                                            implicit_last_nodes.append(pre_pre_node_name)
                                        else:
                                            print("Warning: Special previous nodes sequence for last node {}, "
                                                  "need extra handling!\n".format(node_name))
        if (len(mwnn_last_nodes) - singularity) != len(implicit_last_nodes):
            print("Found valid last nodes for conversion:", implicit_last_nodes)
            print("Origin mwnn last nodes:", mwnn_last_nodes)
            quit(1)
    #print(implicit_last_nodes)

    # start conversion
    layer_count = 0
    for node in graphproto.node:
        node_name = node.name if node.name else node.output[0]
        if node_name in skip_nodes or node_name in ignore_branches:
            continue
        # G.add_node(node_name)
        inputs = []
        for inp in node.input:
            if inp not in initializers:
                # G.add_node(inp)
                # G.add_edge(inp, node.output[0])
                inputs.append(inp)

        outputs = node.output
        print("inputs: ", inputs)
        print("outputs: ", outputs)

        ConverterRegistry.ConverterLayer(
            graphproto, node, node_name, inputs, outputs)
        layer_count += 1

        if node_name in last_nodes or node_name in implicit_last_nodes:
            caffe_last_nodes.append(caffe_network[outputs[0]]['output_name'][3:]) # remove "nn." in name

    print("Totally handled layers count:", layer_count)

    if first_nodes != []:
        for n in ConverterRegistry.caffe_first_nodes:
            caffe_first_nodes.append(n) # note the name is origin framework (before conversion to Caffe layer name)
        print("Set first nodes for Caffe:", caffe_first_nodes)

    if caffe_last_nodes != []:
        print("Set last layers in Caffe:", caffe_last_nodes)
    print("\n")

    nn = caffe.NetSpec()

    for node in caffe_network:
        caffe_layers += caffe_network[node]['functions']
    with open(output_folder+'/caffe_layers.py', 'w') as f:
        f.write(caffe_layers)
    exec(caffe_layers)
    net = nn.to_proto()
    net.name = "mwnn2ev"

    print("Writing the prototxt...")
    with open(output_folder+"/"+mwnn_file+"_convert.prototxt", "w") as f:
        f.write(str(net))

    net = caffe.Net(output_folder+"/"+mwnn_file+"_convert.prototxt", caffe.TEST)
    cnet = caffe_pb2.NetParameter()
    with open(output_folder+"/"+mwnn_file+'_convert.prototxt') as f:
        text_format.Merge(f.read(), cnet)

    print("Writing the caffemodel...")

    for layer in cnet.layer:
        n = layer.name
        k = layer.name.replace(ConverterRegistry.CAFFE_NAME_SLASH, '/')
        k = k.replace(ConverterRegistry.CAFFE_NAME_DOT, '.')
        k = k.replace(ConverterRegistry.CAFFE_NAME_AT, '@')
        k = k.replace(ConverterRegistry.CAFFE_NAME_DASH, '-')
        k = k.replace(ConverterRegistry.CAFFE_NAME_COLON, ':')
        k = k[6:]  # remove "layer_" at beginning
        # for normalization layers
        k = re.sub("/bn_no_scale$", "", k)
        k = re.sub("/mvn_no_scale$", "", k)
        # for extra added Parameter layer
        k = re.sub("/onehot_indices$", "", k)
        k = re.sub("/reshape_input$", "", k)
        k = re.sub("/sigmoid_input$", "", k)
        k = re.sub("/sub_input$", "", k)
        k = re.sub("/mul_input$", "", k)

        if caffe_network.get(k):
            if layer.type in ["Convolution", 'InnerProduct', "Deconvolution"]:
                if 'weight' in caffe_network[k]:
                    net.params[n][0].data.flat = caffe_network[k]["weight"].float_data
                    if 'bias' in caffe_network[k]:
                        net.params[n][1].data.flat = caffe_network[k]['bias'].float_data
            elif layer.type == 'Bias':
                net.params[n][0].data.flat = caffe_network[k]["bias"].float_data

    net.save(output_folder+"/"+mwnn_file+"_convert.caffemodel")
    print("Conversion completed.")
    convert_prototxt = os.path.abspath(output_folder+"/"+mwnn_file+"_convert.prototxt")
    convert_caffemodel = os.path.abspath(output_folder+"/"+mwnn_file+"_convert.caffemodel")
    print("Converted model saved as: "+convert_prototxt+", "+convert_caffemodel)
    return graphproto, convert_prototxt, convert_caffemodel, caffe_last_nodes, caffe_first_nodes
