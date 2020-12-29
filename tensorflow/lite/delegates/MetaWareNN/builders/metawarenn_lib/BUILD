package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "metawarenn_lib",
    srcs = [
        "metawarenn_model.cc",
        "metawarenn_graph.cc",
        "metawarenn_tensor.cc",
        "metawarenn_node.cc",
        "metawarenn_value_info.cc",
    ],
    hdrs = [
        "metawarenn_model.h",
        "metawarenn_graph.h",
        "metawarenn_tensor.h",
        "metawarenn_node.h",
        "metawarenn_element.h",
        "metawarenn_value_info.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/op:metawarenn_ops",
    ],
)
