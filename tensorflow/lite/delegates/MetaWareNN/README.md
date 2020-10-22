## Steps to build TensorFlow-MetaWareNN Delagate

1. Check Bazel version
* `git clone https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-tensorflow.git`
* `git checkout metawarenn_dev`
* Check the bazel version using the command `bazel version`
* If the version is older than 3.1.0, then download bazel version newer than 3.1.0 and install it
* Download and Install bazel 3.6.0 using the following commands:
```
    wget https://github.com/bazelbuild/bazel/releases/download/3.6.0/bazel-3.6.0-installer-linux-x86_64.sh
    chmod +x bazel-3.6.0-installer-linux-x86_64.sh
    ./bazel-3.6.0-installer-linux-x86_64.sh --user
    export PATH="$HOME/bin:$PATH"
```
* Update the version number in `synopsys-tensorflow/.bazelversion` file

2. Configure tensorflow build
```
    cd synopsys-tensorflow
    ./configure
```

3. Create Virtual Environment
```
    sudo pip install virtualenv
    virtualenv --python=/usr/bin/python3.6 /path/to/new/environment
    source /path/to/new/environment/bin/activate
```

4. Build Tensorflow from scratch using below command
```
    bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    pip install /tmp/tensorflow_pkg/tensorflow-2.3.1-cp36-cp36m-linux_x86_64.whl
```

5. Build MetaWareNN and its dependent libraries
```
    bazel build //tensorflow/lite:libtensorflowlite.so //tensorflow/lite/delegates/MetaWareNN/MetaWareNN_lib:MetaWareNN_implementation //tensorflow/lite/delegates/MetaWareNN/builders:model_builder //tensorflow/lite/delegates/MetaWareNN:MetaWareNN_delegate
```


## Run the Inference using MetaWareNN Delegate

1.  Install flatbuffers and set up the include path
```
git clone https://github.com/google/flatbuffers.git
cd flatbuffers
cmake -G "Unix Makefiles"
make

export CPLUS_INCLUDE_PATH=/path/to/synopsys-tensorflow:/path/to/flatbuffers/include
```
    
2.  Set up environment paths with generated MetawareNN dependent libs
```
export LD_LIBRARY_PATH=/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite:/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN:/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN/MetaWareNN_lib:/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders:$LD_LIBRARY_PATH
```

3. Download MobileNet v2 TFlite model
```
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
tar -vzxf mobilenet_v2_1.0_224.tgz
```

4. `cd synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference`  
Open `inference_metawarenn.cpp` and replace the path in line no. 13 with the downloaded MobileNet v2 TFlite model path

5. Compile the inference script  
```
g++ -o inference inference_metawarenn.cpp -I/path/to/flatbuffers/include -L/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite -ltensorflowlite -L/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN -lMetaWareNN_delegate -L/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders -lmodel_builder -L/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN/MetaWareNN_lib -lMetaWareNN_implementation
```

6. Run the object file  
`./inference`
