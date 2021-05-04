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

from .optimize_eltwise import no_broadcast_eltwise_replace
from .optimize_remove_power import remove_power
from .optimize_upsample import resize_replace

def optimize(convert_prototxt, convert_caffemodel, caffe_last_nodes=[]):
    print("\nStart optimization.\n")
    model_path = convert_prototxt[:-9] # remove .prototxt
    optimized_prototxt = model_path + "_optimized.prototxt"
    optimized_caffemodel = model_path + "_optimized.caffemodel"

    generated_prototxt = remove_power(convert_prototxt, optimized_prototxt)
    generated_prototxt = no_broadcast_eltwise_replace(generated_prototxt, optimized_prototxt)
    generated_prototxt, generated_caffemodel = resize_replace(generated_prototxt, convert_caffemodel,
                                                              optimized_prototxt, optimized_caffemodel)

    if generated_prototxt == optimized_prototxt:
        print("Caffe model optimization is done.\n")
    else: # still get the orignal prototxt
        print("No optimization is required.\n")

    return generated_prototxt, generated_caffemodel