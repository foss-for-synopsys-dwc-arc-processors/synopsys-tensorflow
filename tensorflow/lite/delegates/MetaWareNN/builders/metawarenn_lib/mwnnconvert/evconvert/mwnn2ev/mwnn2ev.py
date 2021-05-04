# ==============================================================================
# Copyright 2016  Synopsys, Inc.
#
# This file and the associated documentation are proprietary to Synopsys,
# Inc., and may only be used in accordance with the terms and conditions of
# a written license agreement with Synopsys, Inc.
# Notwithstanding contrary terms in the DFPUC, Licensee may provide the
# binaries of the EV CNN SDK to its end-customer that purchase Licensee ICs
# that incorporate the Synopsys EV processor core with the CNN option,
# subject to confidentiality terms no less restrictive than those contained in
# the DFPUC.  All other use, reproduction, or distribution of this file
# is strictly prohibited.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from .converter import convert
from .optimize import optimize

def mwnn2ev(model, out_dir=os.getcwd(), last_nodes=[], yaml_config=None,
            verify_caffe2=True, verify_onnxruntime=True, input_shape=[], first_nodes=[]):

    os.environ["GLOG_minloglevel"] = '2'

    is_passed = True
    is_passed_optimized = True
    
    if model is not None and os.path.exists(model):
        mwnn_file_abs = os.path.abspath(model)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print("Create the output folder: "+os.path.abspath(out_dir))

    # mwnn2ev conversion
    mwnn_model, convert_prototxt, convert_caffemodel, caffe_last_nodes, caffe_first_nodes = convert(
        model, out_dir, last_nodes, input_shape, first_nodes)

    optimized_prototxt, optimized_caffemodel = optimize(convert_prototxt, convert_caffemodel, caffe_last_nodes)

    if optimized_prototxt != convert_prototxt:
        # only keep the optimized converted model
        os.remove(convert_prototxt)
        if optimized_caffemodel != convert_caffemodel:
            os.remove(convert_caffemodel)