
#!/bin/tcsh
export EV_CNNMODELS_HOME=/ARC/cnn_tools/cnn_models
source /ARC/setup.sh
python /path/to/synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/mwnnconvert/nnac.py --model /path/to/mobilenetv2-7_graphproto.bin --out_dir /path/to/store/nnac_output --placeholder 224,224,3