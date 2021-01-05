package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "metawarenn_kernels_include",
    srcs = [
    ],
    hdrs = [
        "mli_types.h",
        "mli_api.h",
        "mli_check.h",
        "mli_config.h",
        "mli_debug.h",
        "mli_krn_convolution_decl.h",
        "mli_krn_convolution.h",
        "mli_krn_dotprod_decl.h",
        "mli_krn_dotprod_ref.h",
        "mli_krn_dotprod.h",
        "mli_krn_eltwise_decl.h",
        "mli_krn_reduce_sum2d_decl.h",
        "mli_krn_reduce_sum2d_ref.h",
        "mli_krn_reduce_sum2d.h",
        "mli_math_macros.h",
        "mli_math.h",
        "mli_private_types.h",
        "mli_prv_dsp.h",
        "mli_prv_layout.h",
        "mli_prv_load_store.h",
        "mli_prv_quant_decl.h",
        "mli_prv_quant_ref.h",
        "mli_prv_quant.h",
        "mli_prv_tensor.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/kernels/include/api:metawarenn_kernels_api",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/kernels/include/ref:metawarenn_kernels_ref",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/kernels/helpers:mli_helpers",
    ]
)
