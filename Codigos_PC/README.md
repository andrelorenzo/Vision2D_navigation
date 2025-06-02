## MODELS

midas v21 small                 => model-small.onnx
yolo v11 nano                   => yolo11n.onnx
depth anything v2 outdoor small => depth_anything_v2_vits_outdoor_dynamic.onnx
depth pro bnb4                  => model_bnb4.onnx
Midas DPT 384 hybrid            => dpt_hybrid_384.onnx

## REPOS
https://github.com/isl-org/MiDaS
https://github.com/mnjm/yolo11.cpp
https://github.com/imabot2/serialib
https://github.com/fabio-sim/Depth-Anything-ONNX/releases/tag/v2.0.0
https://huggingface.co/onnx-community/DepthPro-ONNX/tree/main/onnx
https://github.com/yan99033/MiDaS-cpp
https://github.com/parkchamchi/MiDaS/releases/tag/22.12.07
https://developer.nvidia.com/cudnn-downloads

## DEPENDENCIES
### C++
opencv 4.12 compile to g++ with cuda enable 
libtorch 2.3-cu121 or libtorch 2.7-cpu
seriallib
yolov11
cuDNN
cmake >= 3.27
  sudo apt purge cmake
  sudo snap install cmake --classic
  cmake --version
  echo 'export PATH=/snap/bin:$PATH' >> ~/.bashrc
  source ~/.bashrc

### python
ultralytics
> Other dependencies are inside each repo as in requirements.txt in MiDaS-cpp


## CUDA acceleration graphics
GPU used: NVidia Geforce MX110 (gama baja) on linux Ubuntu 20.04
Driver: NVIDIA-SMI 550.144.03 (Driver Version: 550.144.03) => sudo apt install nvidia-driver-550
toolkit cuda 12.1 => https://developer.nvidia.com/cuda-12-1-0-download-archive
cuDNN => https://developer.nvidia.com/cudnn-downloads

### Compilation of Opencv 4.12.0-dev 
nvidia-smi = check your GPU and CUDA version cappabilities, if not install => sudo apt install nvidia-driver-550 (or your compaible driver)
nvcc --version => check cuda toolkit version, must be compatible with your driver
(sudo reboot) => optional, if first time installing it is mandatory and must enterin the Enroll MOK, enter your password and enable in it
> install your compatible libtorch and download it in the home folder, must be the cuda compatible version

add this to your ~/.bahrc file:
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

> Must be on 4.x branch for both repos
cd opencv
mkdir build && cd build
sudo apt update
sudo apt install -y build-essential cmake git pkg-config \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
    libdc1394-dev libopenexr-dev libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev libv4l-dev \
    python3-dev python3-numpy libprotobuf-dev protobuf-compiler \
    libgoogle-glog-dev libgflags-dev libgphoto2-dev libeigen3-dev \
    libhdf5-dev doxygen libatlas-base-dev libxine2-dev \
    libopenblas-dev liblapack-dev liblapacke-dev


TO check:
  build type => Release
  cuda = ON
  dnn cuda = ON
  CUDA_ARCH_BIN = 50 (this must be the same architecture as your GPU, you can look it up online)
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D CUDA_ARCH_BIN=50 \
      -D CUDA_ARCH_PTX="" \
      -D ENABLE_FAST_MATH=ON \
      -D WITH_CUBLAS=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_TBB=ON \
      -D BUILD_opencv_python3=OFF \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF \
      ..
make -j$(nproc) => Warning: this will take around 30min-1h
sudo make install

check correct sim links
  grep CUDNN_MAJOR /usr/include/x86_64-linux-gnu/cudnn.h
  => must return:
  #define CUDNN_MAJOR 9
  #define CUDNN_MINOR 1
  #define CUDNN_PATCHLEVEL 0

si no aparece y no existe la carpeta /usr/local/cuda/include
  sudo mkdir -p /usr/local/cuda/include
  sudo ln -sf /usr/include/x86_64-linux-gnu/cudnn.h /usr/local/cuda/include/cudnn.h
  sudo cp include/cudnn*.h /usr/local/cuda/include/
  sudo cp lib/libcudnn* /usr/local/cuda/lib64/
  sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
  sudo cp /usr/include/x86_64-linux-gnu/cudnn_version.h /usr/local/cuda/include/
  sudo cp /usr/include/x86_64-linux-gnu/cudnn.h /usr/local/cuda/include/
VERIFICA => cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR

GET INSTALLATION DETAIL => python3 -c "import cv2; print(cv2.getBuildInformation())"
in case you must reestart
sudo apt purge 'nvidia*' 'cuda*'
sudo apt autoremove