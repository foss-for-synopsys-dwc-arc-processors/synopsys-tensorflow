cd root
mkdir common
cd common

apt -y update
apt -y install git
apt-get -y install wget
apt-get -y install unzip
apt-get -y install openssh-client
apt-get -y install gedit vim
apt-get -y install build-essential
apt-get -y install libssl-dev
apt-get -y install python3-pip
python3 -m pip install --upgrade pip
apt-get -y install zlib1g-dev
apt-get -y install libboost-serialization-dev
apt-get -y install llvm
apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev cmake libedit-dev libxml2-dev libxml2
apt-get -y install graphviz libpng-dev ninja-build wget opencl-headers libgoogle-glog-dev libboost-all-dev libdouble-conversion-dev libevent-dev libssl-dev libgflags-dev libjemalloc-dev libpthread-stubs0-dev
apt-get -y install locales
locale-gen en_US.UTF-8
update-locale LANG=en_US.UTF-8
pip3 install numpy
pip3 install pillow
pip3 install decorator attrs
pip3 install tornado
pip3 install onnx
pip3 install onnxruntime
pip3 install psutil xgboost cloudpickle
pip3 install tflite==2.3.0
pip3 install scipy
pip3 install torch torchvision
pip3 install tensorflow
wget https://github.com/Kitware/CMake/releases/download/v3.16.5/cmake-3.16.5.tar.gz
tar -zxvf cmake-3.16.5.tar.gz
cd cmake-3.16.5
./configure
make
make install
cd ..
wget https://github.com/bazelbuild/bazel/releases/download/3.6.0/bazel-3.6.0-installer-linux-x86_64.sh
chmod +x bazel-3.6.0-installer-linux-x86_64.sh
./bazel-3.6.0-installer-linux-x86_64.sh
git clone https://github.com/google/flatbuffers.git
cd flatbuffers
git checkout v1.12.0
cmake -G "Unix Makefiles"
make
cd ..
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.3/protobuf-all-3.11.3.tar.gz
tar -xf protobuf-all-3.11.3.tar.gz
cd protobuf-3.11.3
./configure
make
make check
make install
cd ./python
python3 setup.py build
python3 setup.py test
python3 setup.py install
export PATH=/usr/local/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/include:${CPLUS_INCLUDE_PATH}
cd ../..
ln -s /usr/bin/python3 /usr/bin/python
git clone https://github.com/fmtlib/fmt
mkdir fmt/build
cd fmt/build
cmake ..
make
make install
cd ../..
cd ..

git clone --recursive https://github.com/SowmyaDhanapal/onnxruntime.git
cd onnxruntime
git checkout metawarenn_dev
git submodule update --init --recursive
cd onnxruntime/core/providers/metawarenn/metawarenn_lib
git checkout onnx_conversion
cd ../../../../..
git add onnxruntime/core/providers/metawarenn/metawarenn_lib
cd ..

git clone --recursive https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-tensorflow.git
cd synopsys-tensorflow
git checkout metawarenn_dev
git submodule update --init --recursive
cd tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib
git checkout onnx_conversion
cd ../../../../../../..

git clone --recursive https://github.com/SowmyaDhanapal/tvm.git tvm
cd tvm
git checkout metawarenn_dev
git submodule sync
git submodule update
git submodule update --init --recursive
cd src/runtime/contrib/metawarenn/metawarenn_lib
git checkout onnx_conversion
cd ../../../../../..

git clone --recursive https://github.com/SowmyaDhanapal/glow.git
cd glow
git checkout metawarenn_dev
git submodule update --init --recursive
cd lib/Backends/MetaWareNN/metawarenn_lib
git checkout onnx_conversion
cd ../../../..
source ./utils/build_llvm.sh
apt install -y clang-6.0
ln -s /usr/bin/clang-6.0 /usr/bin/clang
ln -s /usr/bin/clang++-6.0 /usr/bin/clang++
