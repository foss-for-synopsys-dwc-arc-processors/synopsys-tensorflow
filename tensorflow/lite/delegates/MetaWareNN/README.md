## Steps to build TensorFlow-MetaWareNN Delagate
  
### Use Docker for Installation (optional)
##### Check on the [synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/Docker/README.md](https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-tensorflow/blob/metawarenn_dev/tensorflow/lite/delegates/MetaWareNN/Docker/README.md)  
  
### Common Installation Process
1. Install required bazel version
* Check the bazel version using the command `bazel version`
* Download and Install bazel required 3.6.0 using the following commands:
```
    wget https://github.com/bazelbuild/bazel/releases/download/3.6.0/bazel-3.6.0-installer-linux-x86_64.sh
    chmod +x bazel-3.6.0-installer-linux-x86_64.sh
    ./bazel-3.6.0-installer-linux-x86_64.sh --user
    export PATH="$HOME/bin:$PATH"
```
2. Download & Configure tensorflow build
* `git clone --recursive https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-tensorflow.git`
* `cd synopsys-tensorflow`
* `git checkout metawarenn_dev`

3. MetaWareNN Library submodule setup
* `git submodule update --init --recursive`
*  Move to metawarenn_lib submodule and checkout to metawarenn_dev branch
    a. `cd tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib`
    b. `git checkout metawarenn_dev`
*  Once initial submodule setup is done with above commands, use this command to pull from the submodule in future
    i.  `cd /path/to/synopsys-tensorflow`
    ii. `git pull --recurse-submodules`
```
    cd synopsys-tensorflow
    ./configure
```

4. Create virtual environment and install dependent packages
```
    sudo pip install virtualenv
    virtualenv --python=/usr/bin/python3.6 /path/to/new/environment
    source /path/to/new/environment/bin/activate
    pip install numpy<1.19.0
```

5. Build Tensorflow from scratch
```
    bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
    pip install /tmp/tensorflow_pkg/tensorflow-2.3.1-cp36-cp36m-linux_x86_64.whl
    # Note: 36 could be other numbers if you use other python version
```

#### Run the Inference using MetaWareNN Delegate

1.  Install flatbuffers and set up the include path
```
    git clone https://github.com/google/flatbuffers.git
    cd flatbuffers
    git checkout v1.12.0
    cmake -G "Unix Makefiles"
    make

    export CPLUS_INCLUDE_PATH=/path/to/synopsys-tensorflow:/path/to/flatbuffers/include:${CPLUS_INCLUDE_PATH}
```

2.  Set up environment paths with generated MetawareNN dependent libs
```
  export LD_LIBRARY_PATH=/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite:/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN:/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders:$LD_LIBRARY_PATH
```

3. Download MobileNet v2 TFlite model
```
    wget https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224.tgz
    tar -vzxf mobilenet_v2_1.0_224.tgz
```

4. `cd synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference`  
Open `inference_metawarenn.cpp` and replace the path in line no. 13 with the downloaded MobileNet v2 TFlite model path

5. To Load MetaWareNN Executable Graph in Shared Memory [Default flow]  
   1. Update the "synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/executable_network/metawarenn_executable_graph.cc" with path to store the Executable network binary in line no: 826  
   2. Update the "synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/mwnn_inference_api/mwnn_inference_api.cc" file with saved file path of Executable network binary in line no: 51  

   To Invoke the NNAC & EVGENCNN Script to generate the EV Binary file  
   1. Update the "synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/model_builder.cc" file as follows:  
      i. Set the path to synopsys-tensorflow in line no: 206 & 223  
   2. Update the "synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/model_builder.h" file as follows:  
      i. Set the INVOKE_NNAC macro to 1 in line no: 17  
   3. Update the "synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib/mwnnconvert/mwnn_convert.sh" file as follows:  
      i. Set the $EV_CNNMODELS_HOME path in line no: 3  
      ii. Set the absolute path for ARC/setup.sh file in line no: 4  
      iii. Update the path to sysnopsys-tensorflow with MWNN support in line no: 9 and line no: 20  
      iv. Update the path to evgencnn executable in line no: 10  
      v. Update the Imagenet images path in line no: 18  
      vi. Update evgencnn to evgencnn.pyc if using the release (not development) version of ARC/cnn_tools in line no: 22  
   [Note] : Generated EV Binary file for MetaWareNN SubGraph will store in evgencnn/scripts folder.  

6. Build MetaWareNN dependent libraries  
    * Download protobuf library version 3.11.3 from the egnyte link https://multicorewareinc.egnyte.com/dl/FjljPlgjlI  
    * Unzip and move the "libprotobuf.so" to "/path/to/synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/builders/"    
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
7. Build TFLite with MetaWareNN Delegate Support  
    ```
    bazel build //tensorflow/lite:libtensorflowlite.so //tensorflow/lite/delegates/MetaWareNN/builders:model_builder //tensorflow/lite/delegates/MetaWareNN:MetaWareNN_delegate
    ```

8. Compile the inference script  
  Note: we suggest to use g++ 7 to avoid possible errors.  
    ```
    cd synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference
    g++ -o inference inference_metawarenn.cpp -I/path/to/flatbuffers/include -L/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite -ltensorflowlite -L/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN -lMetaWareNN_delegate -L/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders -lmodel_builder -L/usr/lib/x86_64-linux-gnu -lrt
    ```

9. Run the object file 
  ```
  ipcs # List the shared memory details along with shmid
  ipcrm -m [shmid] # Adjust shared memory allocation size
  ./inference  
  ```

## To run multiple TFLite models from model zoo
   1. `cd /path/to/synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference`
   2. Download the models from TFLite model zoo by running the script
      `sh download_models.sh`
   3. Set the path to synopsys-tensorflow in synopsys-tensorflow/tensorflow/lite/delegates/MetaWareNN/inference/inference_regression.cpp file at line no: 19
   4. Set CPLUS_INCLUDE_PATH and LD_LIBRARY_PATH environment variables as mentioned in previous step 1 & 2
   5. Compile the inference script
      `g++ -o inference inference_regression.cpp -I/path/to/flatbuffers/include -L/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite -ltensorflowlite -L/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN -lMetaWareNN_delegate -L/path/to/synopsys-tensorflow/bazel-bin/tensorflow/lite/delegates/MetaWareNN/builders -lmodel_builder -L/usr/lib/x86_64-linux-gnu -lboost_serialization -L/usr/lib/x86_64-linux-gnu -lrt`
   6. Run the executable
      `./inference`  
      
   Note:  
      i. Invoke call from the inference script currently parses the MetaWareNN Executable graph from shared memory  
