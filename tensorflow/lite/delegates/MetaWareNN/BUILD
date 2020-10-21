load("//tensorflow/lite:build_def.bzl", "tflite_copts", "tflite_linkopts")

package(
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "MetaWareNN_delegate",
    srcs = ["MetaWareNN_delegate_kernel.cc",
            "MetaWareNN_delegate.cc"],
    hdrs = ["MetaWareNN_delegate_kernel.h",
            "MetaWareNN_delegate.h"],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        "//tensorflow/lite:kernel_api",
        "//tensorflow/lite:minimal_logging",
        "//tensorflow/lite:util",
        "//tensorflow/lite/core/api",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates:utils",
        "//tensorflow/lite/kernels:kernel_util",
        "//tensorflow/lite/schema:schema_fbs",
        "//tensorflow/lite/delegates/MetaWareNN/builders:model_builder"
    ],
)
