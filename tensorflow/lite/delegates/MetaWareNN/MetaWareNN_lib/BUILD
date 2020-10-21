package(
    default_visibility = [
        "//visibility:public",
    ],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "MetaWareNN_libs",
    hdrs = [
        "NeuralNetworksTypes.h",
        "NeuralNetworksWrapper.h",
    ],
)

cc_library(
    name = "MetaWareNN_implementation",
    srcs = ["MetaWareNN_implementation.cc"],
    hdrs = [
        "MetaWareNN_implementation.h",
    ],
    tags = [
        "manual",
        "nobuilder",
    ],
    deps = [
        ":MetaWareNN_libs",
    ],
)
