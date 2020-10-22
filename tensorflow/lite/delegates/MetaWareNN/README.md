### Steps to build TensorFlow-MetaWareNN Delagate

* `git clone https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-tensorflow.git`
* `git checkout metawarenn_dev`
* Check the bazel version using the command `bazel version`
* If the version is older than 3.1.0, then download bazel version newer than 3.1.0 and install it
* Download and Install bazel 3.6.0 using below commands:
```
    wget https://github.com/bazelbuild/bazel/releases/download/3.6.0/bazel-3.6.0-installer-linux-x86_64.sh
    chmod +x bazel-3.6.0-installer-linux-x86_64.sh
    ./bazel-3.6.0-installer-linux-x86_64.sh --user
    export PATH="$HOME/bin:$PATH"
```
* Update the version number in `/path/to/synopsys-tensorflow/.bazelversion` file

#### Configure the tensorflow build
```
    cd synopsys-tensorflow
    ./configure
```

#### Create Virtual Environment
```
    sudo pip install virtualenv
    virtualenv --python=/usr//bin/python3.6 /path/to/new/environment
    source /path/to/new/environment/bin/activate
```

#### Build Tensorflow from scratch using below command
```
    bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    pip install /tmp/tensorflow_pkg/tensorflow-2.3.1-cp36-cp36m-linux_x86_64.whl
```

#### Build MetaWareNN and its dependent libraries
```
    bazel build //tensorflow/lite:libtensorflowlite.so //tensorflow/lite/delegates/MetaWareNN/MetaWareNN_lib:MetaWareNN_implementation //tensorflow/lite/delegates/MetaWareNN/builders:model_builder //tensorflow/lite/delegates/MetaWareNN:MetaWareNN_delegate
```
### Run the Inference using MetaWareNN Delegate
1.  Add Include path to flatbuffers and cloned synopsys-tensorflow,
    * export CPLUS_INCLUDE_PATH=/path/to/synopsys-tensorflow:/path/to/flatbuffers/include:
2.  Add Environment Library path with generated MetawareNN dependent libs,
      * export LD_LIBRARY_PATH=/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite:/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN:/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN/MetaWareNN_lib:/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders:$LD_LIBRARY_PATH
3. Download MobilenetV2 model using below command,
    * wget https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
4. cd synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference
5. Open inference_metawarenn.cpp and replace path in line no:12 with the downloaded TFLite model path.
6. Compile the inference script, 
    *   g++ -o inference inference_metawarenn.cpp -I/path/to/synopsys-tensorflow/flatbuffers/include -L/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite -ltensorflowlite -L/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN -lMetaWareNN_delegate -L/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders -lmodel_builder -L/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN/MetaWareNN_lib -lMetaWareNN_implementation
7. Run the object file,
    *   ./inference
