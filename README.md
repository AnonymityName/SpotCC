# [SpotCC]()

## Backend

### install dependency

```shell
sudo apt update
# ubuntu 20
sudo apt install -y openssh-server git curl net-tools build-essential autoconf libtool pkg-config libssl-dev zlib1g-dev libopencv-dev libeigen3-dev tmux wget zip
# ubuntu 22
sudo apt install -y openssh-server git curl net-tools build-essential autoconf libtool pkg-config libssl-dev zlib1g-dev libopencv-dev libeigen3-dev tmux wget zip cmake
```

### cmake 3.22 (ubuntu 20)

```shell
# install cmake with gcc-9
tar -zxvf cmake-3.22.0.tar.gz
cd cmake-3.22.0
./configure
make
sudo make install
```

### gcc-11 (ubuntu 20)

```shell
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 --slave /usr/bin/g++ g++ /usr/bin/g++-11 
```

### gRPC

```shell
# install TRITON_ENABLE_CC_GRPC

# add to ~/.bashrc
export MY_INSTALL_DIR=$HOME/.local 
export PATH="$MY_INSTALL_DIR/bin:$PATH" 
source ~/.bashrc

mkdir -p $MY_INSTALL_DIR

git clone --recurse-submodules -b v1.64.0 --depth 1 --shallow-submodules https://github.com/grpc/grpc

# compile and install grpc
cd grpc
mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
      ../..
make -j 4
sudo make install
popd

# compile test program
cd examples/cpp/helloworld
mkdir -p cmake/build
pushd cmake/build
cmake -DCMAKE_PREFIX_PATH=$MY_INSTALL_DIR ../..
make -j 4

# run test program
./greeter_server
./greeter_client
```

### docker

```shell
# switch to root
# sudo passwd root
su root

# install docker
curl -fsSL http://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | apt-key add -
add-apt-repository "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
# add-apt-repository -r "deb [arch=amd64] http://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
# add-apt-repository "deb [arch=amd64] http://mirrors.tuna.tsinghua.edu.cn/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
apt update
apt install -y docker-ce docker-ce-cli containerd.io
systemctl start docker
```

### triton server

```shell
# Step 1: Create the example model repository
git clone https://github.com/triton-inference-server/server.git
cd server/docs/examples
./fetch_models.sh

# Step 2: Launch triton from the NGC Triton container
# if cuda is available
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
apt update
apt install -y nvidia-docker2
systemctl restart docker
docker run --gpus=1 --rm --net=host -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.06-py3 tritonserver --model-repository=/models

# else if cuda is unavailable
docker run  --rm --net=host -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.06-py3 tritonserver --model-repository=/models

# Step 3: Sending an Inference Request
# In a separate console, launch the image_client example from the NGC Triton SDK container
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:24.06-py3-sdk
/workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg

# Inference should return the following
# Image '/workspace/images/mug.jpg':
#    15.346230 (504) = COFFEE MUG
#    13.224326 (968) = CUP
#    10.422965 (505) = COFFEEPOT
    
# batch request
/workspace/install/bin/image_client -m inception_graphdef -c 3 -s INCEPTION -b 3 /workspace/images

# Request 0, batch size 3
#Image '/workspace/images/mug.jpg':
#    0.754047 (505) = COFFEE MUG
#    0.157065 (969) = CUP
#    0.002878 (968) = ESPRESSO
#Image '/workspace/images/mug.jpg':
#    0.754047 (505) = COFFEE MUG
#    0.157065 (969) = CUP
#    0.002878 (968) = ESPRESSO
#Image '/workspace/images/mug.jpg':
#    0.754047 (505) = COFFEE MUG
#    0.157065 (969) = CUP
#    0.002878 (968) = ESPRESSO
```

### rapidjson

```shell
git clone https://github.com/Tencent/rapidjson.git
cd rapidjson/
mkdir build
cd build
cmake ..
make 
sudo make install
```

## Frontend

### Install Dependency

```shell
sudo apt install -y libtbb-dev
```

### Libtorch

```shell
cd $HOME

# libtorch cpu
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.8.1+cpu.zip

# libtorch gpu, if cuda is available
wget https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcu124.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.6.0+cu124.zip

# add to ~/.bashrc
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$HOME/libtorch/include:$HOME/libtorch/include/torch/csrc/api/include
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$HOME/libtorch/include:$HOME/libtorch/include/torch/csrc/api/include

source ~/.bashrc
```

## Source Compilation

```shell

cd SpotCC
mkdir -p build
cd build
# compile in the first time, need to compile grpc
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/install -DTRITON_ENABLE_CC_GRPC=ON ..
# do not need to compile grpc in the following compiling
cmake -DCMAKE_INSTALL_PREFIX=`pwd`/install -DTRITON_ENABLE_CC_GRPC=OFF ..
make -j 4

# run triton server
# if cuda is available 
docker run --gpus=1 --rm --net=host -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.06-py3 tritonserver --model-repository=/models
# else if cuda is unavailable
docker run  --rm --net=host -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:24.06-py3 tritonserver --model-repository=/models

# run backend
./install/bin/backend_server conf_path

# run frontend
./image_frontend conf_path

# run client
./image_client conf_path data_path
```



根据EuroSys-24，992个GPU的集群，2个月（1440小时）发生100次左右failure，平均每个GPU每小时发生7*10-5次failure
