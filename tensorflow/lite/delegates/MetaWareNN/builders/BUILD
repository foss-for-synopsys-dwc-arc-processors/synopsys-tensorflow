package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "model_builder",
    srcs = [
        "model_builder.cc",
        "op_builder.cc",
        "model.cc",
    ],
    hdrs = [
        "model_builder.h",
        "op_builder.h",
        "helper.h",
        "model.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/MetaWareNN/MetaWareNN_lib:MetaWareNN_implementation",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib:metawarenn_lib",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/optimizer:metawarenn_optimizer",
    ],
)
