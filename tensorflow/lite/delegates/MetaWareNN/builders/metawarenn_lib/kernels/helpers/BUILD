package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "mli_helpers",
    srcs = [
        "mli_helpers.cc",
    ],
    hdrs = [
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/kernels/include:mli_config.h",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/kernels/include:mli_debug.h",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/kernels/include:mli_types.h",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/kernels/include:mli_math.h",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/kernels/include:mli_prv_tensor.h",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/kernels/include:mli_check.h",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/kernels/include:mli_math_macros.h",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/kernels/include:mli_private_types.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/kernels/include/api:metawarenn_kernels_api",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/kernels/include/ref:metawarenn_kernels_ref",
    ],
)
