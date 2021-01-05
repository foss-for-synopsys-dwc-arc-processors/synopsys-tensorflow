package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "metawarenn_kernels",
    srcs = [
    ],
    hdrs = [
        "mli_krn_convolution_ref.h",
        "mli_krn_avepool_hwc.h",
        "mli_krn_eltwise_ref.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/kernels/include:metawarenn_kernels_include",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/kernels/helpers:mli_helpers",
    ],
)
