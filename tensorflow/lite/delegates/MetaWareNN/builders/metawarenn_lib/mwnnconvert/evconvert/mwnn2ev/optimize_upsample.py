#! /usr/bin/python
from __future__ import print_function

from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import caffe
import numpy as np

# Optimize for EV mapping: replace ResizeNearestNeighbor with Deconv layer to improve efficiency

def resize_replace(origin_prototxt, origin_caffemodel, output_prototxt, output_caffemodel):
    net = caffe_pb2.NetParameter()
    with open(origin_prototxt) as f:
        s = f.read()
    txtf.Merge(s, net)

    run_net = caffe.Net(origin_prototxt, origin_caffemodel, caffe.TEST)
    #run_net.forward()

    new_deconv_layers = []

    optimized = False
    for l in net.layer:
        if l.type == "ResizeNearestNeighbor":
            num_output = run_net.blobs[l.name].shape[1]
            # TODO: check the other cases
            if l.resize_nearest_neighbor_param.HasField("half_pixel_onnx"):
                if l.resize_nearest_neighbor_param.half_pixel_onnx == True:
                    continue
            scale_height = l.resize_nearest_neighbor_param.scale_height
            scale_width = l.resize_nearest_neighbor_param.scale_width
            # Deconv layer could not hanlde floating cases
            if int(scale_height) - scale_height != 0:
                continue
            else:
                scale_height = int(scale_height)
            if int(scale_width) - scale_width !=0:
                continue
            else:
                scale_width = int(scale_width)
            print("ResizeNearestNeighbor layer {} is replaced by Deconvolution layer.".format(l.name))
            l.type = "Deconvolution"
            l.convolution_param.num_output = num_output
            l.convolution_param.group = num_output
            if scale_height == scale_width:
                l.convolution_param.kernel_size.append(scale_height)
                l.convolution_param.stride.append(scale_height)
            else:
                l.convolution_param.kernel_h = scale_height
                l.convolution_param.kernel_w = scale_width
                l.convolution_param.stride_h = scale_height
                l.convolution_param.stride_W = scale_width
            #l.convolution_param.pad.append(0)
            l.convolution_param.bias_term = False
            l.convolution_param.weight_filler.type = "constant"
            l.convolution_param.weight_filler.value = 1
            l.ClearField("resize_nearest_neighbor_param")
            optimized = True
            new_deconv_layers.append(l.name)

    if optimized:
        print('Writing updated prototxt to {}.'.format(output_prototxt))
        with open(output_prototxt, 'w') as f:
            f.write(str(net))
        f.close()

        # Add the Deconv weights in new caffemodel
        run_net_new = caffe.Net(output_prototxt, caffe.TEST)
        for l in new_deconv_layers:
            shape = run_net_new.params[l][0].data.shape
            run_net_new.params[l][0].data[...] = np.ones(shape)
        # Copy the other weights and bias from original caffemodel
        for N in run_net.params.keys():
            for i in range(len(run_net.params[N])):
                run_net_new.params[N][i].data[...] = run_net.params[N][i].data
        run_net_new.save(output_caffemodel)
        print('Writing updated caffemodel to {}.\n'.format(output_caffemodel))
        #run_net_new.forward()
        return output_prototxt, output_caffemodel
    else:
        print("No optimization is needed for Upsample.\n")
        return origin_prototxt, origin_caffemodel

#resize_replace("/home/yche/cnn_tools/evgencnn/scripts/evconvert/yolov3_input_dim_fixed_convert_optimized.prototxt",
#               "/home/yche/cnn_tools/evgencnn/scripts/evconvert/yolov3_input_dim_fixed_convert.caffemodel",
#               "/home/yche/cnn_tools/evgencnn/scripts/evconvert/yolov3_input_dim_fixed_convert_updated.prototxt",
#               "/home/yche/cnn_tools/evgencnn/scripts/evconvert/yolov3_input_dim_fixed_convert_updated.caffemodel")