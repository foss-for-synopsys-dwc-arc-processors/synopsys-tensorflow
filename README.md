# Synopsys Tensorflow
It is a modified version of the popular [TensorFlow](https://github.com/tensorflow/tensorflow) for use with Synopsys DesignWare EV Processors 
and support EV style of quantization.  
The origin is based on TensorFlow 1.14.0.  

## Installation guide

### Linux
#### Prerequisites
* Python - 3.6
* Cuda - 10.0
* Cudnn - 7.6.2

#### Environment Setup  
* Bazel Installation
    * Download bazel [bazel-0.24.1-installer-linux-x86_64.sh](https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-installer-linux-x86_64.sh)
    * Install the prerequisites: `sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python`
    * `chmod +x bazel-0.24.1-installer-linux-x86_64.sh`
    * `./bazel-0.24.1-installer-linux-x86_64.sh --user`
    * export PATH="$HOME/bin:$PATH"

* Install Python and the TensorFlow package dependencies
    * `sudo apt install python3.6-dev python3-pip`
    * `python3.6 -m pip install -U --user pip six numpy wheel setuptools mock`
    * `python3.6 -m pip install -U --user keras_applications==1.0.6 --no-deps`
    * `python3.6 -m pip install -U --user keras_preprocessing==1.0.5 --no-deps`
    * `sudo apt install python3-numpy`

* Download the TensorFlow source code
    * `git clone https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-tensorflow.git`
    * `cd synopsys-tensorflow`
    * `./configure` (To configure the GPU build, sample as follows)
    ```
        * Please specify the location of python. [Default is /usr/bin/python]: /usr/bin/python3.6
        * Found possible Python library paths:
            /usr/lib/python3/dist-packages
            /usr/local/lib/python3.6/dist-packages
        * Please input the desired Python library path to use.  Default is [/usr/lib/python3/dist-packages]
        * Do you wish to build TensorFlow with XLA JIT support? [Y/n]: n
            No XLA JIT support will be enabled for TensorFlow.
        * Do you wish to build TensorFlow with OpenCL SYCL support? [y/N]: N
            No OpenCL SYCL support will be enabled for TensorFlow.
        * Do you wish to build TensorFlow with ROCm support? [y/N]: N
            No ROCm support will be enabled for TensorFlow.
        * Do you wish to build TensorFlow with CUDA support? [y/N]: y
            CUDA support will be enabled for TensorFlow.
        * Do you wish to build TensorFlow with TensorRT support? [y/N]: N
            No TensorRT support will be enabled for TensorFlow.

            Found CUDA 10.0 in:
                /usr/local/cuda/lib64
                /usr/local/cuda/include
            Found cuDNN 7 in:
                /usr/local/cuda/lib64
                /usr/local/cuda/include

        * Please specify a list of comma-separated CUDA compute capabilities you want to build with.
            You can find the compute capability of your device at: https://developer.nvidia.com/cuda-gpus.
            Please note that each additional compute capability significantly increases your build time and binary size, and that TensorFlow only supports compute capabilities >= 3.5 [Default is: 6.1]:
        * Do you want to use clang as CUDA compiler? [y/N]: N
            nvcc will be used as CUDA compiler.
        * Please specify which gcc should be used by nvcc as the host compiler. [Default is /usr/bin/gcc]:
        * Do you wish to build TensorFlow with MPI support? [y/N]: N
            No MPI support will be enabled for TensorFlow.
        * Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is -march=native -Wno-sign-compare]:
        * Would you like to interactively configure ./WORKSPACE for Android builds? [y/N]: N
            Not configuring the WORKSPACE for Android builds.

        Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
        	--config=mkl         	# Build with MKL support.
        	--config=monolithic  	# Config for mostly static monolithic build.
        	--config=gdr         	# Build with GDR support.
        	--config=verbs       	# Build with libverbs support.
        	--config=ngraph      	# Build with Intel nGraph support.
        	--config=numa        	# Build with NUMA support.
    	    --config=dynamic_kernels	# (Experimental) Build kernels into separate shared objects.
        Preconfigured Bazel build configs to DISABLE default on features:
        	--config=noaws       	# Disable AWS S3 filesystem support.
        	--config=nogcp       	# Disable GCP support.
        	--config=nohdfs      	# Disable HDFS support.
        	--config=noignite    	# Disable Apache Ignite support.
        	--config=nokafka     	# Disable Apache Kafka support.
        	--config=nonccl      	# Disable NVIDIA NCCL support.
    Configuration finished
    ````

* Virtual Environment Setup (optional)
    * `sudo pip install virtualenv`
    * `virtualenv --python=/usr/bin/python3.6 /path/to/new/environment`
    * `source /path/to/new/environment/bin/activate`

* Build the package
    * `bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package`
    * `./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg`
    * `pip install /tmp/tensorflow_pkg/tensorflow-1.14.1-cp36-cp36m-linux_x86_64.whl`

* Install other dependencies (optional) 
    * `pip install keras==2.2.4`
    

### Windows
#### Prerequisites
* Python - 3.6
* Cuda - 10.0 (optional)
* Cudnn - 7.6.2 (optional)

#### Environment Setup  
* Install MSYS2
  * Download the installer at https://www.msys2.org/ and follow the installation steps there
  * If MSYS2 is installed to `C:\msys64`, add `C:\msys64\usr\bin` to your `%PATH%` environment variable. 
  * Open `cmd.exe`, run `pacman -S git patch unzip`

* Install Visual C++ Build Tools 2019
   * Open https://visualstudio.microsoft.com/zh-hans/downloads/
   * Select `Redistributables and Build Tools`, download and install `Microsoft Visual C++ 2019 Redistributable` and `Microsoft Build Tools 2019` (optional, if you have already installed previous version)

* Bazel Installation
    * Download bazel [bazel-0.24.1-windows-x86_64.exe](https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel-0.24.1-windows-x86_64.exe)
    * Install the prerequisites [Visual C++ Redistributable for Visual Studio 2015](https://www.microsoft.com/en-us/download/details.aspx?id=48145)
    * Rename the Bazel binary to `bazel.exe` and add the folder that contains it to your `%PATH%` environment variable.
    * Run `bazel version` to ensure that you have successfully installed it

* Install Python package dependencies
    * `pip install six numpy wheel`
    * `pip install keras_applications==1.0.6 --no-deps`
	 * `pip install keras_preprocessing==1.0.5 --no-deps`

* Download the TensorFlow source code
    * `git clone https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-tensorflow.git`
    * `cd synopsys-tensorflow`
    * `python configure.py` (To configure the CPU only build, sample as follows)
```
Please specify the location of python. [Default is C:\Users\yche\AppData\Local\Programs\Python\Python36\python.exe]:

Found possible Python library paths:
  C:\Users\yche\AppData\Local\Programs\Python\Python36\lib\site-packages
  \Users\yche\Desktop\Projects\synopsys-caffe\python
Please input the desired Python library path to use.  Default is [C:\Users\yche\AppData\Local\Programs\Python\Python36\lib\site-packages]

Do you wish to build TensorFlow with XLA JIT support? [y/N]: N
No XLA JIT support will be enabled for TensorFlow.

Do you wish to build TensorFlow with ROCm support? [y/N]: N
No ROCm support will be enabled for TensorFlow.

Do you wish to build TensorFlow with CUDA support? [y/N]: N
No CUDA support will be enabled for TensorFlow.

Please specify optimization flags to use during compilation when bazel option "--config=opt" is specified [Default is /arch:AVX]:

Would you like to override eigen strong inline for some C++ compilation to reduce the compilation time? [Y/n]: n
Not overriding eigen strong inline, some compilations could take more than 20 mins.

Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
        --config=mkl            # Build with MKL support.
        --config=monolithic     # Config for mostly static monolithic build.
        --config=gdr            # Build with GDR support.
        --config=verbs          # Build with libverbs support.
        --config=ngraph         # Build with Intel nGraph support.
        --config=numa           # Build with NUMA support.
        --config=dynamic_kernels        # (Experimental) Build kernels into separate shared objects.
Preconfigured Bazel build configs to DISABLE default on features:
        --config=noaws          # Disable AWS S3 filesystem support.
        --config=nogcp          # Disable GCP support.
        --config=nohdfs         # Disable HDFS support.
        --config=noignite       # Disable Apache Ignite support.
        --config=nokafka        # Disable Apache Kafka support.
        --config=nonccl         # Disable NVIDIA NCCL support.
````

* Virtual Environment Setup (optional)
    * `pip install virtualenv`
    * `virtualenv --python=python3.6 env_path`
    * `.\env_path\Scripts\activate.bat`

* Build the package
    * `bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package`
    * `bazel-bin\tensorflow\tools\pip_package\build_pip_package C:/tmp/tensorflow_pkg`
    * `pip install C:/tmp/tensorflow_pkg/tensorflow-1.14.1-cp36-cp36m-win_amd64.whl`

* Install other dependencies (optional) 
    * `pip install keras==2.2.4`
    
Other detailed installation instructions can be found at [official TensorFlow guideline](https://www.tensorflow.org/install/source_windows).
