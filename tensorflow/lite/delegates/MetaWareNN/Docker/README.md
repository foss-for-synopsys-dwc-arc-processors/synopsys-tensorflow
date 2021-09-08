## Steps to use the docker setup to build and run the TensorFlow
1. To create a docker container with Ubuntu 18.04 as base, run  
        * `sudo bash Docker.sh`  
2. Copy the shell script to docker folder,   
        * `scp uname@ip_address:/path/to/local/machine/tf_deps.sh /path/to/docker/folder`
3. Run the shell script to install the TF related dependencies,  
        * `cd /path/to/docker/folder`
        * `bash tf_deps.sh`  
        [Note]: The above commands will install all TF related dependencies including, bazel, protobuf, flatbuffers, etc., and clones the synopsys-tensorflow repository. It will take more than an hour to finish the installation.  
4. To build the TensorFlow,
        * `cd /path/to/synopsys-tensorflow/`
        * `bazel build //tensorflow/lite:libtensorflowlite.so //tensorflow/lite/delegates/MetaWareNN/builders:model_builder //tensorflow/lite/delegates/MetaWareNN:MetaWareNN_delegate`  

### To run the Inference,
Changes in the TF source code needs the last command(`bazel build*`) of step-4, to rebuild the code  
* ### To Load MetaWareNN Executable Graph in Shared Memory [Default flow]
   1. Set the path to synopsys-tensorflow in tensorflow/lite/delegates/MetaWareNN/inference/env.sh line no: 5
* ### To Invoke the NNAC & EVGENCNN Script to generate the EV Binary file
   1. Enable INVOKE_NNAC in synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/model_builder.h line no: 25
   2. Update synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference/env.sh file
      i. Set the path to ARC/ directory in lino no: 11
      ii. Set the path to cnn_models/ directory in lino no: 12
   [Note] : Generated EV Binary file for MetaWareNN SubGraph will store in evgencnn/scripts folder and all intermediate files will get stored in `/path/to/synopsys-tensorflow/NNAC_DUMPS` folder

### To Run Inference using MetaWareNN Backend  
* #### Download the MobileNet-V2 model using the following link,  
    *   wget https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz  
    *   tar -vzxf mobilenet_v2_1.0_224.tgz  
* `scp uname@ip_address:/path/to/local/machine/mobilenet_v2_1.0_224.tflite /path/to/docker/folder/`  
* Update the docker path of mobilenet_v2_1.0_224.tflite in `/path/to/docker/TFLite/synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference/inference_metawarenn.cpp` in line no: 13  
* `cd synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference/`
* Set the path to flatbuffers in synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference/env.sh line no:16
* `source env.sh`
* `g++ -o inference inference_metawarenn.cpp -I$FLATBUFFERS_PATH/include -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite -ltensorflowlite -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN -lMetaWareNN_delegate -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders -lmodel_builder -L/usr/lib/x86_64-linux-gnu -lrt`
* `./inference`