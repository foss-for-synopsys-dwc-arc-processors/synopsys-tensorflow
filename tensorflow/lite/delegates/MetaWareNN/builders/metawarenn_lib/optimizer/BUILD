package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "metawarenn_optimizer",
    srcs = [
        "metawarenn_optimizer.cc",
        "dummy_pass_1.cc",
        "remove_reshape.cc",
        "fuse_relu.cc",
        "pass_manager.cc",
    ],
    hdrs = [
        "metawarenn_optimizer.h",
        "dummy_pass_1.h",
        "remove_reshape.h",
        "fuse_relu.h",
        "pass_manager.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib:metawarenn_lib",
    ],
)
