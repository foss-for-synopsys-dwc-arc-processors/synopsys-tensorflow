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


from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

# To remove the "pass-by" power layer after conversion

def remove_power(origin_prototxt, output_prototxt):
    net = caffe_pb2.NetParameter()
    with open(origin_prototxt) as f:
        s = f.read()
    txtf.Merge(s, net)

    optimized = False
    for l in net.layer:
        if l.type == 'Power':
            if l.HasField("power_param"):
                continue
            pName = l.name
            bottom = l.bottom[0]
            top = l.top[0] # For the case when top name not equals to layer name
            for L in net.layer:
                for i in range(len(L.bottom)):
                    if L.bottom[i] == pName and L.top[0] == pName:
                        L.bottom[i] = bottom
                        L.top[0] = bottom
                    if L.bottom[i] == pName and L.top[0] != pName:
                        L.bottom[i] = bottom
                    if L.bottom[i] == top:
                        L.bottom[i] = bottom
            net.layer.remove(l)
            optimized = True
            print("Remove {}".format(pName))

    # Double check
    for l in net.layer:
        if l.type == 'Power':
            if l.HasField("power_param"):
                continue
            pName = l.name
            bottom = l.bottom[0]
            for L in net.layer:
                for i in range(len(L.bottom)):
                    if L.bottom[i] == pName and L.top[0] == pName:
                        L.bottom[i] = bottom
                        L.top[0] = bottom
                    elif L.bottom[i] == pName and L.top[0] != pName:
                        L.bottom[i] = bottom
                    else:
                        pass
            net.layer.remove(l)
            optimized = True
            print("Remove {}".format(pName))


    if optimized:
        print('Writing to {}.\n'.format(output_prototxt))
        with open(output_prototxt, 'w') as f:
            f.write(str(net))
        f.close()
        return output_prototxt
    else:
        print("No optimization is needed for Power.\n")
        return origin_prototxt

#remove_power("/home/yche/cnn_tools/evgencnn/scripts/evconvert/detectron2_retinanet_R_101_FPN_3_convert_updated.prototxt",
#             "/home/yche/cnn_tools/evgencnn/scripts/evconvert/detectron2_retinanet_R_101_FPN_3_convert_updated.prototxt")
