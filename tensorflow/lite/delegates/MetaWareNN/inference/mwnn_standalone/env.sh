#!/bin/sh

export MWNN_STANDALONE_PATH=/path/to/mwnn_standalone/
export METAWARENN_LIB_PATH=$MWNN_STANDALONE_PATH/metawarenn_lib/
export PROTOBUF_PATH=$METAWARENN_LIB_PATH/lib/libprotobuf.so
export EXEC_DUMPS_PATH=$MWNN_STANDALONE_PATH"/EXEC_DUMPS/"