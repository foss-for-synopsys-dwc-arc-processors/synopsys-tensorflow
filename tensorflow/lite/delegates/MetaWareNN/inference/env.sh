#!/bin/sh

########### Executable Networks flow ##############
#set the path to synopsys-tensorflow
export FRAMEWORK_PATH=/Path/to/synopsys-tensorflow/
export METAWARENN_LIB_PATH=$FRAMEWORK_PATH"/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/"
export EXEC_DUMPS_PATH=$FRAMEWORK_PATH"/EXEC_DUMPS/"

########### NNAC - EV binary generation flow ##############
#set the path to ARC directory
export ARC_PATH=/path/to/ARC/
export NNAC_DUMPS_PATH=$FRAMEWORK_PATH"/NNAC_DUMPS/"

########### Common Library Path Settings ################
export FLATBUFFERS_PATH=/Path/to/flatbuffers/
export CPLUS_INCLUDE_PATH=$FRAMEWORK_PATH:$FLATBUFFERS_PATH/include

export LD_LIBRARY_PATH=$FRAMEWORK_PATH"/bazel-bin/tensorflow/lite":$FRAMEWORK_PATH"/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders":$FRAMEWORK_PATH"/bazel-bin/tensorflow/lite/delegates/MetaWareNN":$LD_LIBRARY_PATH