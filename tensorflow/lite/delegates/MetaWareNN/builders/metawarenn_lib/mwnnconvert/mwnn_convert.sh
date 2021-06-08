
#!/bin/tcsh
export EV_CNNMODELS_HOME=/path/to/ARC/cnn_tools/cnn_models
source /path/to/ARC/setup.sh
bin_path=$1
op_path=$2
graph_name=$3
subgraph_counter=$4
python /path/to/synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/mwnnconvert/nnac.py --model $bin_path --out_dir $op_path"/nnac_op" --placeholder 224,224,3
cd /path/to/ARC/cnn_tools/utils/tools/evgencnn/scripts
prototxt_path=$op_path"/nnac_op/"$graph_name"/"$graph_name"_convert.prototxt"
caffe_model_path=$op_path"/nnac_op/"$graph_name"/"$graph_name"_convert.caffemodel"
FILE=$op_path"/nnac_op/"$graph_name"/"$graph_name"_convert_optimized.prototxt"
if [ -f "$FILE" ]; then
    prototxt_path=$FILE
fi
if [ "$subgraph_counter" = "1" ]; then
  input_path="/path/to/cnn_models/caffe_models/images/ImageNet/jpeg"
else
  input_path="/path/to/synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/input/pkl"
fi
python3.6 evgencnn  --evss_cfg ev_native --evss_debug 0 --debug_error_log --report_verbose --id 1  --cnn_srcdir $op_path"/evgencnn/cnn_src/"$graph_name --outdir $op_path"/evgencnn/models/"$graph_name --name $graph_name --caffe $prototxt_path --weights $caffe_model_path --images $input_path --generator host_fixed