package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "metawarenn_kernels",
    srcs = [
        "helpers/mli_helpers.cc",
    ],
    hdrs = [
        "mli_krn_convolution_ref.h",
        "mli_krn_avepool_hwc.h",
        "mli_krn_eltwise_ref.h",
        "include/mli_types.h",
        "include/mli_api.h",
        "include/mli_check.h",
        "include/mli_config.h",
        "include/mli_debug.h",
        "include/mli_krn_convolution_decl.h",
        "include/mli_krn_convolution.h",
        "include/mli_krn_dotprod_decl.h",
        "include/mli_krn_dotprod_ref.h",
        "include/mli_krn_dotprod.h",
        "include/mli_krn_eltwise_decl.h",
        "include/mli_krn_reduce_sum2d_decl.h",
        "include/mli_krn_reduce_sum2d_ref.h",
        "include/mli_krn_reduce_sum2d.h",
        "include/mli_math_macros.h",
        "include/mli_math.h",
        "include/mli_private_types.h",
        "include/mli_prv_dsp.h",
        "include/mli_prv_layout.h",
        "include/mli_prv_load_store.h",
        "include/mli_prv_quant_decl.h",
        "include/mli_prv_quant_ref.h",
        "include/mli_prv_quant.h",
        "include/mli_prv_tensor.h",
        "include/ref/mli_math.h",
        "include/api/mli_helpers_api.h",
        "include/api/mli_kernels_api.h",
        "include/api/mli_krn_avepool_spec_api.h",
        "include/api/mli_krn_conv2d_spec_api.h",
        "include/api/mli_krn_depthwise_conv2d_spec_api.h",
        "include/api/mli_krn_maxpool_spec_api.h",
        "include/api/mli_mov_api.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
)
