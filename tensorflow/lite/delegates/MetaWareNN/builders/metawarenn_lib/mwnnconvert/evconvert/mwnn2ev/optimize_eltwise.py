#! /usr/bin/python
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

from __future__ import print_function

from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import caffe

# When no broadcasting happens, replace Add/Mul with Eltwise SUM/PROD to increase efficiency

def no_broadcast_eltwise_replace(origin_prototxt, output_prototxt):
    net = caffe_pb2.NetParameter()
    with open(origin_prototxt) as f:
        s = f.read()
    txtf.Merge(s, net)

    run_net = caffe.Net(origin_prototxt, caffe.TEST)
    #run_net.forward()

    optimized = False
    for l in net.layer:
        if l.type == 'Add':
            b0 = l.bottom[0]
            b1 = l.bottom[1]
            if run_net.blobs[b1].data.shape == run_net.blobs[b0].data.shape:
                print("No broadcasting occurs. Add layer {} is replaced by Eltwise SUM layer.".format(l.name))
                l.type = "Eltwise"
                l.eltwise_param.operation = 1 #SUM
                optimized = True

        if l.type == 'Mul':
            b0 = l.bottom[0]
            b1 = l.bottom[1]
            if run_net.blobs[b1].data.shape == run_net.blobs[b0].data.shape:
                print("No broadcasting occurs. Mul layer {} is replaced by Eltwise PROD layer.".format(l.name))
                l.type = "Eltwise"
                l.eltwise_param.operation = 0 #PROD
                optimized = True


    if optimized:
        print('Writing to {}.\n'.format(output_prototxt))
        with open(output_prototxt, 'w') as f:
            f.write(str(net))
        f.close()
        return output_prototxt
    else:
        print("No optimization is needed for Eltwise.\n")
        return origin_prototxt

#no_broadcast_eltwise_replace(origin_prototxt="/home/yche/cnn_tools/evgencnn/scripts/evconvert/detectron2_retinanet_R_101_FPN_3_convert_updated.prototxt",
#                             output_prototxt="/home/yche/cnn_tools/evgencnn/scripts/evconvert/detectron2_retinanet_R_101_FPN_3_convert_optimized.prototxt")