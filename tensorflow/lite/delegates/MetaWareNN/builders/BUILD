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
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib:optimizer/metawarenn_optimizer.cc",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib:optimizer/remove_reshape.cc",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib:optimizer/fuse_relu.cc",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib:optimizer/convert_layout.cc",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib:optimizer/pass_manager.cc",
    ],
    hdrs = [
        "model_builder.h",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib:optimizer/metawarenn_optimizer.h",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib:optimizer/remove_reshape.h",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib:optimizer/fuse_relu.h",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib:optimizer/convert_layout.h",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib:optimizer/pass_manager.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:util",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib:metawarenn_lib",
    ],
)
