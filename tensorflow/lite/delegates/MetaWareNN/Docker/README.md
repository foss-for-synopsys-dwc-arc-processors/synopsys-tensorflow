## Steps to use the docker setup to build and run the TensorFlow
1. To create a docker container with Ubuntu 18.04 as base, run
    * `sudo bash Docker.sh`
2. Copy the shell script to docker folder,
    * `cp /path/to/local/machine/tf_deps.sh /path/to/docker/folder/root`
3. Run the shell script to install the TF related dependencies,
    * `cd /path/to/docker/folder/root`
    * `bash tf_deps.sh`
       * [Note]: The above commands will install all TF related dependencies including, bazel, protobuf, flatbuffers, etc., and clones the synopsys-tensorflow repository. It will take more than an hour to finish the installation.
4. Modifications to make before build,
    * Download the dependent protobuf library from egnyte link https://multicorewareinc.egnyte.com/dl/kpRzPTSFdx and place it in `synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/lib`
5. To build the TensorFlow,
    * `cd /path/to/synopsys-tensorflow/`
    * `./configure`
    * `bazel build //tensorflow/lite:libtensorflowlite.so //tensorflow/lite/delegates/MetaWareNN/builders:model_builder //tensorflow/lite/delegates/MetaWareNN:MetaWareNN_delegate`

### To Run the Inference,
Changes in the TF source code needs the last command(`bazel build*`) of step-5, to rebuild the code
* ### To Load MetaWareNN Executable Graph in Shared Memory [Default flow]
   1. Set the path to synopsys-tensorflow in tensorflow/lite/delegates/MetaWareNN/inference/env.sh line no: 5
   2. Set the path to flatbuffers in tensorflow/lite/delegates/MetaWareNN/inference/env.sh line no:16
* ### To Invoke the NNAC & EVGENCNN Script to generate the EV Binary file
       1. Enable INVOKE_NNAC in synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/model_builder.h line no: 25
       2. Update synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference/env.sh file
         i. Set the path to ARC/ directory in lino no: 11
        ii. Set the path to cnn_models/ directory in lino no: 12
              * [Note] : Generated EV Binary file for MetaWareNN SubGraph will store in evgencnn/scripts folder and all intermediate files will get stored in `/path/to/synopsys-tensorflow/NNAC_DUMPS` folder

### Set the Environmental Variables for Inference
         * `source /path/to/docker/synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference/env.sh`

### To Run Inference using MetaWareNN Backend
* #### Download the MobileNet-V2 model using the following link,
    *   wget https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
    *   tar -vzxf mobilenet_v2_1.0_224.tgz
* Update the docker path of mobilenet_v2_1.0_224.tflite in `/path/to/docker/TFLite/synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference/inference_metawarenn.cpp` in line no: 13
* `cd synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference/`
* `g++ -o inference inference_metawarenn.cpp -I$FLATBUFFERS_PATH/include -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite -ltensorflowlite -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN -lMetaWareNN_delegate -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders -lmodel_builder -L/usr/lib/x86_64-linux-gnu -lrt`
* `./inference`

### To Run Multiple TFLite models from model zoo
   1. `cd $FRAMEWORK_PATH/tensorflow/lite/delegates/MetaWareNN/inference`
   2. Download the models from TFLite model zoo by running the script
      `sh download_models.sh`
   3. Compile the inference script
      `g++ -o inference inference_regression.cpp -I$FLATBUFFERS_PATH/include -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite -ltensorflowlite -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN -lMetaWareNN_delegate -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders -lmodel_builder -L/usr/lib/x86_64-linux-gnu -lrt`
   4. Run the executable
      `./inference`

### To Generate the ONNXProto from multiple TFLite models & Verify
   1. `cd tensorflow/lite/delegates/MetaWareNN/inference`
   2. `source env.sh`
   3. `sh download_models.sh` # (For First time) - Creates tflite_models directory inside synopsys-tensorflow/ & downloads models into it
   4. Compile the inference script
      `g++ -o inference inference_metawarenn.cpp -I$FLATBUFFERS_PATH/include -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite -ltensorflowlite -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN -lMetaWareNN_delegate -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders -lmodel_builder -L/usr/lib/x86_64-linux-gnu -lrt`
   5. `python3 test_regression_tflite.py` # Creates a `op_tflite_models` directory and dump the generated ONNXProto files for all input models & `validation_result.txt` file which contains the comparison of original tflite & generated onnx model

### To Generate the ONNXProto from multiple Quantized TFLite models & Verify
   1. `cd tensorflow/lite/delegates/MetaWareNN/inference`
   2. `source env.sh`
   3. `sh download_quantized_models.sh` # (For First time) - Creates tflite_quantized_models directory inside synopsys-tensorflow/ & downloads models into it
   4. Compile the inference script
      `g++ -o inference inference_metawarenn.cpp -I$FLATBUFFERS_PATH/include -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite -ltensorflowlite -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN -lMetaWareNN_delegate -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders -lmodel_builder -L/usr/lib/x86_64-linux-gnu -lrt`
   5. `python3 test_regression_quantized_tflite.py` # Creates a `op_tflite_quantized_models` directory and dump the generated ONNXProto files for all input models & `validation_result.txt` file which contains the comparison of original tflite & generated onnx model
