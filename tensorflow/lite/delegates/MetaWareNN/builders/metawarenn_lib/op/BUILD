package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "metawarenn_ops",
    srcs = [
        "add.cc",
        "avg_pool.cc",
        "constant.cc",
        "conv.cc",
        "depthwise_conv.cc",
        "input_data.cc",
        "node.cc",
        "relu.cc",
        "reshape.cc",
        "softmax.cc",
    ],
    hdrs = [
        "add.h",
        "avg_pool.h",
        "constant.h",
        "conv.h",
        "depthwise_conv.h",
        "input_data.h",
        "node.h",
        "relu.h",
        "reshape.h",
        "softmax.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib:metawarenn_element",
    ],
)
