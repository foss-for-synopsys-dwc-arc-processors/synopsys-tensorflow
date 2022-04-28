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
   ### To create ONNX Proto from MWNNGraph by loading TFLite Models[Default flow]
   1. By default, `INFERENCE_ENGINE` flag is set to zero in metawarenn_lib/metawarenn_common.h, which will create ONNXProto directly from MWNNGraph and store it in inference/op_onnx_models
   2. Enable `INFERENCE_ENGINE` flag in metawarenn_lib/metawarenn_common.h, to convert MWNNGraph to ExecutableGraph and then create Inference Engine & Execution Context and finally creates the output ONNXProto in inference/op_onnx_models for model verification
   ### To Invoke the NNAC & EVGENCNN Script to generate the EV Binary file - Outdated & Optional [Not tested after MWNNGraph update to ONNX format]
   1. Enable INVOKE_NNAC in tensorflow/lite/delegates/MetaWareNN/builders/model_builder.h line no: 22
   2. Update tensorflow/lite/delegates/MetaWareNN/inference/env.sh file
      i. Set the path to ARC/ directory in lino no: 11
      ii. Set the path to cnn_models/ directory in lino no: 12
  ```
   [Note] : Generated EV Binary file for MetaWareNN SubGraph will store in evgencnn/scripts folder and all intermediate files will get stored in `/path/to/synopsys-tensorflow/NNAC_DUMPS` folder
  ```
   ### To Use metawarenn_lib as Shared Library - Outdated & Optional
   1. Rename tensorflow/lite/delegates/MetaWareNN/builders/BUILD to BUILD_original
      * `mv tensorflow/lite/delegates/MetaWareNN/builders/BUILD tensorflow/lite/delegates/MetaWareNN/builders/BUILD_original`
   2. Rename tensorflow/lite/delegates/MetaWareNN/builders/BUILD_shared_lib to BUILD
      * `mv tensorflow/lite/delegates/MetaWareNN/builders/BUILD_shared_lib tensorflow/lite/delegates/MetaWareNN/builders/BUILD`
   3. Download the metawarenn shared library from egnyte link https://multicorewareinc.egnyte.com/dl/n31afFTwP9 and place it in `synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/lib`
   4. Also download the dependent protobuf library from egnyte link https://multicorewareinc.egnyte.com/dl/kpRzPTSFdx and place it in `synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/lib`

### Set the Environmental Variables for Inference
         * `source /path/to/docker/synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference/env.sh`

### To Run Inference using MetaWareNN Backend
* #### Download the MobileNet-V2 model using the following egnyte link,
    *   https://multicorewareinc.egnyte.com/dl/USS5sw1FZG
* `cd synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference/`
* `g++ -o inference inference_metawarenn.cpp -I$FLATBUFFERS_PATH/include -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite -ltensorflowlite -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN -lMetaWareNN_delegate -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders -lmodel_builder -L/usr/lib/x86_64-linux-gnu -lrt`
* Run the inference
      * To run Float model - `./inference /path/to/float/tflite/model float`
      * To run Quantized model - `./inference /path/to/quant/tflite/model uint8_t`

### To Generate the ONNXProto from multiple Float TFLite models & Verify
   1. `cd tensorflow/lite/delegates/MetaWareNN/inference`
   2. `source env.sh`
   3. `sh download_models.sh` or Download float TFLite Models from Egnyte Link - https://multicorewareinc.egnyte.com/fl/k0wHUistvX # (For First time) - Creates tflite_models directory inside synopsys-tensorflow/ & downloads models into it
   4. Compile the inference script
      `g++ -o inference inference_metawarenn.cpp -I$FLATBUFFERS_PATH/include -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite -ltensorflowlite -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN -lMetaWareNN_delegate -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders -lmodel_builder -L/usr/lib/x86_64-linux-gnu -lrt`
   5. `python3 test_regression_tflite.py` # Creates a `op_tflite_models` directory and dump the generated ONNXProto files for all input models & `validation_result.txt` file which contains the comparison of original tflite & generated onnx model

### To Generate the ONNXProto from multiple Quantized TFLite models & Verify
   1. `cd tensorflow/lite/delegates/MetaWareNN/inference`
   2. `source env.sh`
   3. `sh download_quantized_models.sh` or Download Quantized TFLite Models from Egnyte Link - https://multicorewareinc.egnyte.com/fl/7uaWWI9PNi # (For First time) - Creates tflite_quantized_models directory inside synopsys-tensorflow/ & downloads models into it
   4. Compile the inference script
      `g++ -o inference inference_metawarenn.cpp -I$FLATBUFFERS_PATH/include -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite -ltensorflowlite -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN -lMetaWareNN_delegate -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders -lmodel_builder -L/usr/lib/x86_64-linux-gnu -lrt`
   5. `python3 test_regression_quantized_tflite.py` # Creates a `op_tflite_quantized_models` directory and dump the generated ONNXProto files for all input models & `validation_result.txt` file which contains the comparison of original tflite & generated onnx model
