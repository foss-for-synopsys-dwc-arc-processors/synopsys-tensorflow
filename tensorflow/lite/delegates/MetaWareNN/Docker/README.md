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
        * `scp uname@ip_address:/path/to/local/machine/lib_protobuf_MWNN_PROTO.zip /path/to/docker/folder/TFLite/synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders`
        * `cd /path/to/docker/folder/TFLite/synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders`
        * `unzip /path/to/docker/folder/TFLite/synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/lib_protobuf_MWNN_PROTO.zip`
        * `cd /path/to/synopsys-tensorflow/`
        * `bazel build //tensorflow/lite:libtensorflowlite.so //tensorflow/lite/delegates/MetaWareNN/builders:model_builder //tensorflow/lite/delegates/MetaWareNN:MetaWareNN_delegate`

### To run the Inference,
Changes in the TF source code needs the last command(`bazel build*`) of step-4, to rebuild the code
* #### To Load MetaWareNN Executable Graph in Shared Memory[Default flow]
    1. Update the "synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/executable_network/metawarenn_executable_graph.cc" with path to store the Executable network binary in line no: 756
    2. Update the "synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/mwnn_inference_api/mwnn_inference_api.cc" file with saved file path of Executable network binary in line no: 51
* #### To Invoke the NNAC & EVGENCNN Script to generate the EV Binary file
    1. Update the "synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/model_builder.cc" file as follows:
        i. Set the path to store the MWNN file dumps in line no: 200 
        ii. Set the path to synopsys-tensorflow in line no: 209
    2. Update the "synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/model_builder.h" file as follows: 
        i. Set the INVOKE_NNAC macro to 1 in line no: 17
    3. Update the "synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/mwnnconvert/mwnn_convert.sh" file as follows: 
        i. Set the $EV_CNNMODELS_HOME path in line no: 3 
        ii. Set the absolute path for ARC/cnn_tools/setup.sh file in line no: 4 
        iii. Update the path to synopsys-tensorflow with MWNN support in line no: 9 and line no: 20 
        iv. Update the path to evgencnn executable in line no: 10 
        v. Update the Imagenet images path in line no: 18
        [Note] : Generated EV Binary file for MetaWareNN SubGraph will store in evgencnn/scripts folder.

### To Run Inference using MetaWareNN Backend
* #### Download the MobileNet-V2 model using the following link,
    *   wget https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
    *   tar -vzxf mobilenet_v2_1.0_224.tgz
* `scp uname@ip_address:/path/to/local/machine/mobilenet_v2_1.0_224.tflite /path/to/docker/folder/`
* `Update the docker path of mobilenet_v2_1.0_224.tflite in /path/to/docker/TFLite/synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference/inference_metawarenn.cpp in line no: 13`
* `export CPLUS_INCLUDE_PATH=/path/to/docker/TFLite/synopsys-tensorflow:/path/to/docker/TFLite/flatbuffers/include`
* `export LD_LIBRARY_PATH=/path/to/docker/TFLite/synopsys-tensorflow/bazel-bin/tensorflow/lite:/path/to/docker/TFLite/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN:/path/to/docker/TFLite/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders:$LD_LIBRARY_PATH`
* `cd tensorflow/lite/delegates/MetaWareNN/inference/`
* `g++ -o inference inference_metawarenn.cpp -I/path/to/docker/TFLite/flatbuffers/include -L/path/to/docker/TFLite/synopsys-tensorflow/bazel-bin/tensorflow/lite -ltensorflowlite -L/path/to/docker/TFLite/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN -lMetaWareNN_delegate -L/path/to/docker/TFLite/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders -lmodel_builder -L/usr/lib/x86_64-linux-gnu -lrt`
