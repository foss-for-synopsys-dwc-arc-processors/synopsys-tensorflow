mkdir TFLite
cd TFLite
apt -y update
apt -y install git
apt-get install openssh-client
apt-get install vim
apt-get -y install wget
apt-get -y install unzip
apt-get -y install gedit
apt-get -y install build-essential
apt-get -y install libssl-dev
wget https://github.com/bazelbuild/bazel/releases/download/3.6.0/bazel-3.6.0-installer-linux-x86_64.sh
chmod +x bazel-3.6.0-installer-linux-x86_64.sh
./bazel-3.6.0-installer-linux-x86_64.sh
wget https://github.com/Kitware/CMake/releases/download/v3.16.5/cmake-3.16.5.tar.gz
tar -zxvf cmake-3.16.5.tar.gz
cd cmake-3.16.5
./configure
make
make install
wget https://github.com/git-lfs/git-lfs/releases/download/v2.13.3/git-lfs-linux-amd64-v2.13.3.tar.gz
tar -xf git-lfs-linux-amd64-v2.13.3.tar.gz
chmod 755 install.sh
./install.sh
cd ..
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
apt-get -y install python3-distutils
apt-get -y install python3-apt
apt-get -y install python3-pip
python3 setup.py build
python3 setup.py test
python3 setup.py install
export PATH=/usr/local/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/include:${CPLUS_INCLUDE_PATH}
cd ../..
pip3 install numpy
 ln -s /usr/bin/python3 /usr/bin/python
git clone https://github.com/foss-for-synopsys-dwc-arc-processors/synopsys-tensorflow.git
cd synopsys-tensorflow
git checkout metawarenn_dev
git submodule update --init --recursive
cd tensorflow/lite/delegates/MetaWareNN/builders/metawarenn_lib
git checkout metawarenn_dev
git lfs install
git lfs pull