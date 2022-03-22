## Steps to build TensorFlow-MetaWareNN Delagate

### Use Docker for Installation (optional)
##### Check on the [synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/Docker/README.md](https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-tensorflow/blob/metawarenn_dev/tensorflow/lite/delegates/MetaWareNN/Docker/README.md)

### Prerequisites

  #### Initial Setup
    * `git clone --recursive https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-tensorflow.git`
    * `cd synopsys-tensorflow`
    * `git checkout metawarenn_dev`
    * Use below commands to pull MetaWareNN Library Submodule
        * `git pull`
        * `git submodule update --init --recursive`
        *  Move to metawarenn_lib submodule and checkout to onnx_conversion branch
            a. `cd tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib`
            b. `git checkout onnx_conversion`
  #### Using Existing Setup to pull submodule changes[Docker / Non-Docker]
    * `cd synopsys-tensorflow && git pull`
    * `cd tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib`
    * `git checkout onnx_conversion`
    * `git pull`

  #### Install required bazel version
    * Check the bazel version using the command `bazel version`
    * Download and Install bazel required 3.6.0 using the following commands:
  ```
        wget https://github.com/bazelbuild/bazel/releases/download/3.6.0/bazel-3.6.0-installer-linux-x86_64.sh
        chmod +x bazel-3.6.0-installer-linux-x86_64.sh
        ./bazel-3.6.0-installer-linux-x86_64.sh --user
        export PATH="$HOME/bin:$PATH"
  ```

  #### Build Protobuf Library
    * Required Protobuf Version - 3.11.3, Check with the following command,
      $ protoc --version
    * Install Protobuf version 3.11.3 with below set of commands
  ```
        wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.3/protobuf-all-3.11.3.tar.gz
        tar -xf protobuf-all-3.11.3.tar.gz
        cd protobuf-3.11.3
        ./configure [--prefix=install_protobuf_folder]
        make
        make check
        sudo make install
        cd ./python
        python3 setup.py build
        python3 setup.py test
        sudo python3 setup.py install
        sudo ldconfig
        # if not installed with sudo
        export PATH=install_protobuf_folder/bin:${PATH}
        export LD_LIBRARY_PATH=install_protobuf_folder/lib:${LD_LIBRARY_PATH}
        export CPLUS_INCLUDE_PATH=install_protobuf_folder/include:${CPLUS_INCLUDE_PATH}
  ```
  #### Install flatbuffers and set up the include path
  ```
        git clone https://github.com/google/flatbuffers.git
        cd flatbuffers
        git checkout v1.12.0
        cmake -G "Unix Makefiles"
        make
  ```

  #### Create virtual environment and install dependent packages
  ```
        sudo pip install virtualenv
        virtualenv --python=/usr/bin/python3.6 /path/to/new/environment
        source /path/to/new/environment/bin/activate
        pip install numpy<1.19.0
  ```
#### Configure tensorflow
* cd synopsys-tensorflow
* ./configure

### Download MobileNet v2 TFlite model
  ```
    wget https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
    tar -vzxf mobilenet_v2_1.0_224.tgz
  ```
#### Modifications to make before build
  ```
   1. Download the dependent protobuf library from egnyte link https://multicorewareinc.egnyte.com/dl/kpRzPTSFdx and place it in `synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/lib
  ```
#### To create ONNX Proto from MWNNGraph by loading TFLite Models[Default flow]
   1. By default, `INFERENCE_ENGINE` flag is set to zero in metawarenn_lib/metawarenn_common.h, which will create ONNXProto directly from MWNNGraph and store it in inference/op_onnx_models
   2. Enable `INFERENCE_ENGINE` flag in metawarenn_lib/metawarenn_common.h, to convert MWNNGraph to ExecutableGraph and then create Inference Engine & Execution Context and finally creates the output ONNXProto in inference/op_onnx_models for model verification
   ### To Invoke the NNAC & EVGENCNN Script to generate the EV Binary file - Outdated [Not tested after MWNNGraph update to ONNX format]
   1. Enable INVOKE_NNAC in tensorflow/lite/delegates/MetaWareNN/builders/model_builder.h line no: 22
   2. Update tensorflow/lite/delegates/MetaWareNN/inference/env.sh file
      i. Set the path to ARC/ directory in lino no: 11
      ii. Set the path to cnn_models/ directory in lino no: 12
  ```
   [Note] : Generated EV Binary file for MetaWareNN SubGraph will store in evgencnn/scripts folder and all intermediate files will get stored in `/path/to/synopsys-tensorflow/NNAC_DUMPS` folder
  ```
   ### To Use metawarenn_lib as Shared Library - Outdated
   1. Rename tensorflow/lite/delegates/MetaWareNN/builders/BUILD to BUILD_original
      `mv tensorflow/lite/delegates/MetaWareNN/builders/BUILD tensorflow/lite/delegates/MetaWareNN/builders/BUILD_original`
   2. Rename tensorflow/lite/delegates/MetaWareNN/builders/BUILD_shared_lib to BUILD
      `mv tensorflow/lite/delegates/MetaWareNN/builders/BUILD_shared_lib tensorflow/lite/delegates/MetaWareNN/builders/BUILD`
   3. Download the metawarenn shared library from egnyte link https://multicorewareinc.egnyte.com/dl/n31afFTwP9 and place it in `synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/lib`
   4. Also download the dependent protobuf library from egnyte link https://multicorewareinc.egnyte.com/dl/kpRzPTSFdx and place it in `synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/lib`

### Build TFLite with MetaWareNN Delegate Support
```
bazel build //tensorflow/lite:libtensorflowlite.so //tensorflow/lite/delegates/MetaWareNN/builders:model_builder //tensorflow/lite/delegates/MetaWareNN:MetaWareNN_delegate
```

### Compile and run the inference script
   Note: we suggest to use g++ 7 to avoid possible errors.
   1. `cd tensorflow/lite/delegates/MetaWareNN/inference`
   2. Set the path to flatbuffers in tensorflow/lite/delegates/MetaWareNN/inference/env.sh line no:15
   3. `source env.sh`
   4. Set the path to downloaded MobileNet v2 TFlite model in `synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference/inference_metawarenn.cpp` line no: 13
   5. `g++ -o inference inference_metawarenn.cpp -I$FLATBUFFERS_PATH/include -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite -ltensorflowlite -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN -lMetaWareNN_delegate -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders -lmodel_builder -L/usr/lib/x86_64-linux-gnu -lrt`
   6. Run the inference
      i. To run Float model - `./inference /path/to/tflite/model float` 
      ii. To run Quantized model - `./inference /path/to/tflite/model uint8_t` 

### To Generate the ONNXProto from multiple TFLite models & Verify
   1. `cd tensorflow/lite/delegates/MetaWareNN/inference`
   2. `source env.sh`
   3. `sh download_models.sh` # (For First time) - Creates tflite_models directory inside synopsys-tensorflow/ & downloads models into it
   4. Compile the inference script
      `g++ -o inference inference_metawarenn.cpp -I$FLATBUFFERS_PATH/include -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite -ltensorflowlite -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN -lMetaWareNN_delegate -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders -lmodel_builder -L/usr/lib/x86_64-linux-gnu -lrt`
   5. `python test_regression_tflite.py` # Creates a `op_tflite_models` directory and dump the generated ONNXProto files for all input models & `validation_result.txt` file which contains the comparison of original tflite & generated onnx model

### To Generate the ONNXProto from multiple Quantized TFLite models & Verify
   1. `cd tensorflow/lite/delegates/MetaWareNN/inference`
   2. `source env.sh`
   3. `sh download_quantized_models.sh` # (For First time) - Creates tflite_quantized_models directory inside synopsys-tensorflow/ & downloads models into it
   4. Compile the inference script
      `g++ -o inference inference_metawarenn.cpp -I$FLATBUFFERS_PATH/include -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite -ltensorflowlite -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN -lMetaWareNN_delegate -L$FRAMEWORK_PATH/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders -lmodel_builder -L/usr/lib/x86_64-linux-gnu -lrt`
   5. `python test_regression_quantized_tflite.py` # Creates a `op_tflite_quantized_models` directory and dump the generated ONNXProto files for all input models & `validation_result.txt` file which contains the comparison of original tflite & generated onnx model