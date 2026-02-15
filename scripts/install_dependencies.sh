sudo apt update -y 
sudo  apt-get install build-essential autoconf libtool pkg-config -y
sudo apt install protobuf-compiler libprotobuf-dev -y
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:longsleep/golang-backports -y
sudo apt-get install libjemalloc-dev -y
sudo apt update
sudo apt install cmake -y
cd ..
# install abseil
mkdir Source && cd Source
git clone https://github.com/abseil/abseil-cpp.git
cd abseil-cpp
mkdir build && cd build
cmake -DABSL_BUILD_TESTING=ON -DABSL_USE_GOOGLETEST_HEAD=ON -DCMAKE_CXX_STANDARD=14 ..
make -j$(nproc)
sudo make install

man protoc

cd ../..
# install grpc
git clone -b v1.67.0 https://github.com/grpc/grpc --recursive
cd grpc
git submodule update --init
# sudo apt install apt-transport-https curl gnupg -y
# curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
# sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
# echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
# sudo apt update && sudo apt install bazel -y
# sudo apt update && sudo apt full-upgrade -y
# bazel build :all --enable_bzlmod=false
mkdir -p cmake/build
cd cmake/build
cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local/ ../..
make -j$(nproc)
sudo make install
cd ../../../..

cd d-HNSW
#!/bin/bash

sudo apt update
sudo apt-get install libc-ares-dev -y
sudo apt install protobuf-compiler -y
# Install GKlib
echo "Installing GKlib..."
cd deps

cd GKlib
make config
make -j$(nproc)
sudo make install
cd ..

# Install METIS
echo "Installing METIS..."

cd METIS
make config shared=1 prefix=~/local
make -j$(nproc)
sudo make install
cd ..

# Install gflags
echo "Installing gflags..."

cd gflags
mkdir -p build && cd build
ccmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
sudo make install
cd ..

# Install gRPC
# echo "Installing gRPC..."

# cd grpc
# git submodule update --init
# mkdir -p cmake/build
# cd cmake/build
# # cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF ../..
# cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local/ ../..
# make -j$(nproc)
# sudo make install

# cd ../

cd ../../


sudo apt install libgrpc++-dev libgrpc-dev -y

echo "Dependency installation completed."
