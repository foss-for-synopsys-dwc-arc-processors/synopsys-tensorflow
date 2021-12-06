download_dir=$FRAMEWORK_PATH"/tflite_quantized_models"
mkdir $download_dir
cd $download_dir
wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz
tar -xvf mobilenet_v1_1.0_224_quant.tgz --one-top-level
rm mobilenet_v1_1.0_224_quant.tgz
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz
tar -xvf mobilenet_v2_1.0_224_quant.tgz --one-top-level
rm mobilenet_v2_1.0_224_quant.tgz
wget https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_224_quant_20181026.tgz
tar -xvf inception_v1_224_quant_20181026.tgz --one-top-level
rm inception_v1_224_quant_20181026.tgz
wget https://storage.googleapis.com/download.tensorflow.org/models/inception_v2_224_quant_20181026.tgz
tar -xvf inception_v2_224_quant_20181026.tgz --one-top-level
rm inception_v2_224_quant_20181026.tgz
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz
tar -xvf inception_v3_quant.tgz --one-top-level
rm inception_v3_quant.tgz
wget https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_299_quant_20181026.tgz
tar -xvf inception_v4_299_quant_20181026.tgz
rm inception_v4_299_quant_20181026.tgz --one-top-level