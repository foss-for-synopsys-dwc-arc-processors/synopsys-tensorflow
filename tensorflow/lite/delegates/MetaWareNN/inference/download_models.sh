download_dir=$FRAMEWORK_PATH"/tflite_models"
mkdir $download_dir
cd $download_dir
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/densenet_2018_04_27.tgz
tar -xvf densenet_2018_04_27.tgz
rm densenet_2018_04_27.tgz
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/squeezenet_2018_04_27.tgz
mkdir squeezenet_2018_04_27
tar -xvf squeezenet_2018_04_27.tgz -C squeezenet_2018_04_27
rm squeezenet_2018_04_27.tgz
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_mobile_2018_04_27.tgz
mkdir nasnet_mobile_2018_04_27
tar -xvf nasnet_mobile_2018_04_27.tgz -C nasnet_mobile_2018_04_27
rm nasnet_mobile_2018_04_27.tgz
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/nasnet_large_2018_04_27.tgz
mkdir nasnet_large_2018_04_27
tar -xvf nasnet_large_2018_04_27.tgz -C nasnet_large_2018_04_27
rm nasnet_large_2018_04_27.tgz
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz
mkdir resnet_v2_101
tar -xvf resnet_v2_101.tgz -C resnet_v2_101
rm resnet_v2_101.tgz
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v3_2018_04_27.tgz
mkdir inception_v3_2018_04_27
tar -xvf inception_v3_2018_04_27.tgz -C inception_v3_2018_04_27
rm inception_v3_2018_04_27.tgz
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_v4_2018_04_27.tgz
mkdir inception_v4_2018_04_27
tar -xvf inception_v4_2018_04_27.tgz -C inception_v4_2018_04_27
rm inception_v4_2018_04_27.tgz
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_zoo/upload_20180427/inception_resnet_v2_2018_04_27.tgz
mkdir inception_resnet_v2_2018_04_27
tar -xvf inception_resnet_v2_2018_04_27.tgz -C inception_resnet_v2_2018_04_27
rm inception_resnet_v2_2018_04_27.tgz
wget https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
mkdir mobilenet_v1_1.0_224
tar -xvf mobilenet_v1_1.0_224.tgz -C mobilenet_v1_1.0_224
rm mobilenet_v1_1.0_224.tgz
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
mkdir mobilenet_v2_1.0_224
tar -xvf mobilenet_v2_1.0_224.tgz -C mobilenet_v2_1.0_224
rm mobilenet_v2_1.0_224.tgz