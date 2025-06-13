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
> Other dependencies are inside each repo as in requirements.txt for MiDaS-cpp


## CUDA acceleration graphics
GPU used: NVidia Geforce MX110 (gama baja) on linux Ubuntu 20.04
Driver: NVIDIA-SMI 550.144.03 (Driver Version: 550.144.03) => sudo apt install nvidia-driver-550
toolkit cuda 12.1 => https://developer.nvidia.com/cuda-12-1-0-download-archive
cuDNN => https://developer.nvidia.com/cudnn-downloads

### Compilation of Opencv 4.12.0-dev with cuda enable
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

'''
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
'''
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



#### GET PARAMS CAMERA

Calibración completada. RMS error = 0.452364
Matriz de cámara:
[600.7834767433284, 0, 327.8484673353018;
 0, 598.7307346090644, 242.2022189077919;
 0, 0, 1]
Coeficientes de distorsión:
[-6.946764169443098e-05, -0.3069070976245156, 0.005136360789573193, 0.002372243746375077, 0.9915232424793156]


Calibración completada. RMS error = 0.393792
Matriz de cámara:
[599.510444503993, 0, 329.7388389292479;
 0, 600.6529976678057, 249.8465416157918;
 0, 0, 1]
Coeficientes de distorsión:
[-0.009965509512882243, -0.4090656448452808, 0.007196740517235716, 0.00170554008373496, 1.868935953586357]


Calibración completada. RMS error = 0.652295
Matriz de cámara:
[633.5609082650195, 0, 282.8529257448194;
 0, 626.4882498736677, 181.2755050671355;
 0, 0, 1]
Coeficientes de distorsión:
[-0.02226462996368151, 0.2145592061875851, -0.01592862671585067, -0.02082322436440881, -0.2101272620235353]


Calibración completada. RMS error = 0.153897
Matriz de cámara:
[1429.687426535897, 0, 2.345779198844677;
 0, 1496.305230510821, 0.5525587808315574;
 0, 0, 1]
Coeficientes de distorsión:
[0.02292376645176789, 1.823863900337936, -0.04059310729020604, -0.03962620930370201, -5.611120547335047]


Calibración completada. RMS error = 0.209138
Matriz de cámara:
[1903.801706267169, 0, 19.79555923437522;
 0, 1629.041100231063, -17.67457558094145;
 0, 0, 1]
Coeficientes de distorsión:
[0.0831825813662391, 6.331596730412845, -0.05031060093059213, -0.1361903246428663, -28.69305476525752]


Calibración completada. RMS error = 0.46543
Matriz de cámara:
[1110.791496776183, 0, 149.7155237823486;
 0, 1592.093428030067, -207.6245010017638;
 0, 0, 1]
Coeficientes de distorsión:
[0.1133762527018083, 0.3292021252494029, -0.008388537824247139, -0.02453424393820355, -3.747646221354791]


Calibración completada. RMS error = 0.319242
Matriz de cámara:
[1958.071739244158, 0, -4.894081690110388;
 0, 2070.268483151062, -29.07563571012702;
 0, 0, 1]
Coeficientes de distorsión:
[0.1484825458947004, 22.15475878691031, 0.02662463314626885, -0.2211813814511062, -172.8111262721066]


Calibración completada. RMS error = 0.467588
Matriz de cámara:
[614.203848723158, 0, 335.0001486844355;
 0, 615.0309826791437, 231.9911857583894;
 0, 0, 1]
Coeficientes de distorsión:
[0.003020203552268366, -0.3320056921193665, 0.000168496196883253, 0.005193955718590672, 2.272181445086679]


 Media de la matriz de cámara (fx, fy, cx, cy)

Calculando el promedio:
fx=599.510+1429.687+1903.802+1958.072/4=1472.768
fy=600.653+1496.305+1629.041+2070.268/4=1449.567
cx=329.739+2.346+19.796−4.894/4=86.747
cy=249.847+0.553−17.675−29.076/4=50.412

# en mm
F_x​=1472.768⋅0.00421875≈6.21 mm
F_y​=1449.567⋅0.004375≈6.34 mm
C_x​=86.747⋅0.00421875≈0.37 mm
C_y​=50.412⋅0.004375≈0.22 mm

[0.061656, 7.475788, -0.014270, -0.098824, -51.811591]