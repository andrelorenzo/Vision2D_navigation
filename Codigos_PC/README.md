# Windows support 
In this README, you will find a walktrough on what I did to be able to compile and run this code with CUDA, Cudnn and Opencv, it is not guaranted that will work on you too but it may give you some hints on how to do it.

> [!NOTE]
> Here you will not find any documentations, repos or information on models, algorithms and versions as this information has been added in the Linux-support branch (main).

### My parameters
* Windows 11 Home
* Compiler: MSVC 2022 vc17 (VS Comunity installed is higly recommended): https://visualstudio.microsoft.com/es/downloads/
* CMake (>) 3.25 (GUI installed is highly recommended): https://cmake.org/download/
* Pyhton (>) 3.5: https://www.python.org/downloads/
* (Opcional) Gstreamer 1.26 (msvc): https://gstreamer.freedesktop.org/download/#windows 
* CUDA 12.9 : https://developer.nvidia.com/cuda-downloads
* libtorch 2.3-cu128 (windows) : https://pytorch.org/get-started/locally/
* cuDNN 9.10.2 (for CUDA 12.9): https://developer.nvidia.com/cudnn-downloads
* OpenCV: git clone https://github.com/opencv/opencv.git (YOU must git checkout 4.10.0)
* OPenCV modules: git clone https://github.com/opencv/opencv_contrib.git (YOU must git checkout 4.10.0)
* GPU: NVIDIA GeForce RTX 2050 (laptop) with CUDA_ARCH = 8.6
* GPU driver: 570

### Installing CUDA
You need to install CUDA first, be carefull at the end, it may not have installed everything you need and no error message will be display (only in the end window will show you what did not got installed).

#### EVERYTHING INSTALLED
If everything got installed correctly, you are the luckiest person alive (congratulations), keep reading!!!.

#### NOT EVERYTHING INSTALLED
If not everything got installed, just keep calm, it happens to the best of us, for me the solution was to install a different version of CUDA (11.8) and in the installer only select the things that the other one did not managed to get installed (in my case the nvToolsExt)
It will probably install it on other directory, just copy the contents of **lib, bin and include** into the corresponding folders in the 12.9 directory (mine was C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9 but it may be other for you).


### Installing cuDNN
For this just install cuDNN and in the installer select your cuda version (12.9).

> [!TIP]
> Copy the contents of **lib, bin and include** into the corresponding folders in the 12.9 directory (mine was C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9 but it may be other for you). This is not mandatory but it will simplify things when compiling with CMake as it will probably find the right paths for you.

### Installing libtorch
For this is quite simple, just download and unzip the folder somewhere you will rememember (I put them in C:\libtorch).

### Installing Gstreamer (Completely optional)
I compiled with gstreamer because I now i will use it but IT IS NOT NECESARY, just install both .msi files (dev and runtime) and keep them somewhere you will remember (I put them in C:\gstreamer).

### Installing OpenCV
for this just do (in my case in C:\opencv_cuda):

```bash
cd C:\
mkdir opencv_cuda
cd opencv_cuda

git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.10.0

git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 4.10.0

cd ..
mkdir build
```
We will get the source from opencv, the extra modules form opencv_contrib and we will put the build in build/.


### Checking installitions and enviromental variables
Now you should have everything you need, but first lets confirm the installations, open a CMD and type:

```bash
nvidia-smi
nvcc --version
```

* On `nvidia-smi` should appear with your GPU, GPU driver version and CUDA version (for me GeForce RTX 205, Driver: 570 and CUDA: 12.9).
* On `nvcc --version`should appear a message showing that the toolkit is detected with the correct CUDA version (12.9.xx for me). 
* Click on windows key and search for `Enviromental variables` then `enviroment varibles -> Path -> add the following:`
  * bin OpenCV dir: `C:\opencv\build\x64\vc17\bin` => only if you plan to cmopile opencv with dynamic linking enable
  * bin CUDA dir: `C:\Program FIles\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin` => mandatory
  * bin libtorch dir: `C:\libtorch\lib` => mandatory
  * exe msvc dir: `C:\Program FIles\Microsft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64` => mandatory
  * bin cudnn dir: `C:\Program FIles\NVIDIA\CUDNN\v9.10\bin` => mandatory

  * bin gstreamer dir `C:\gstreamer\1.0\msvc_x8664\bin`=> only if you install gstreamer
  * Finally you have to add a new enviromental variable (not in PATH) but next to it with :
    * Name: CudaToolkitDir
    * Value: `C:\Program FIles\NVIDIA GPU Computing Toolkit\CUDA\v12.9`
> [!WARNING]
> This are my personal path, for you may and will be different.

### Compiling OpenCV
* Execute cmake-gui:
* Source code: C:\opencv_cuda\opencv
* Where to build: C:\opencv\build
* Press configure and select: Visual Studio 17 2022, x64, then click finish

Then it comes the fun (not so fun) part of selecting, deselecting and inputting paths, for some of this configuration will not appear until you configure multiple times =>Example: when you click `WITH_CUDA` and click configure a new set of parameters will appear that have to do with CUDA, this will happen with everything so you must configure and check again all the parameters until everything is setup, also you must click the `advanced` button in the right up corner at the right of search.
Finally, some of this may be already covered for you (if you followed the steps most of the paths will be already okay)

>[!TIP]
>Also check that the paths you  copy exists in your PC, as this can give you a hint on what is missing!

```cmake
CMAKE_BUILD_TYPE: Release
BUILD_SHARED_LIBS: OFF <- satic compilation
CMAKE_INSTALL_PREFIX: C:/build
OPENCV_EXTRA_MODULES_PATH: C:/opencv_cuda/opencv_contrib/modules

WITH_CUDA: ON
CUDA_TOOLKIT_ROOT_DIR: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9
CUDA_TOOLKIT_INCLUDE: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include
WITH_CUDNN: ON
OPENCV_DNN_CUDA: ON
CUDNN_LIBRARY: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib (if you copy the cudnn files into CUDA, if not must point to the folder with cudnn.lib)
CUDNN_INCLUDE_DIR: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include (same as library)

BUILD_opencv_dnn: ON
ENABLE_FAST_MATH: ON
CUDA_FAST_MATH: ON
WITH_TBB: OFF

BUILD_TESTS: OFF
BUILD_PERF_TESTS: OFF
BUILD_DOCS: OFF
BUILD_EXAMPLES: OFF (optional)

WITH_GSTREAMER: ON (optional)
OPENCV_GAPI_GSTREAMER: OFF
GSTREAMER_app_LIBRARY	            C:/gstreamer/1.0/msvc_x86_64/lib/gstapp-1.0.lib
GSTREAMER_audio_LIBRARY	          C:/gstreamer/1.0/msvc_x86_64/lib/gstaudio-1.0.lib
GSTREAMER_base_LIBRARY	          C:/gstreamer/1.0/msvc_x86_64/lib/gstbase-1.0.lib
GSTREAMER_glib_INCLUDE_DIR	      C:/gstreamer/1.0/msvc_x86_64/include/glib-2.0
GSTREAMER_glib_LIBRARY	          C:/gstreamer/1.0/msvc_x86_64/lib/glib-2.0.lib
GSTREAMER_glibconfig_INCLUDE_DIR	C:/gstreamer/1.0/msvc_x86_64/lib/glib-2.0/include
GSTREAMER_gobject_LIBRARY	        C:/gstreamer/1.0/msvc_x86_64/lib/gobject-2.0.lib
GSTREAMER_gst_INCLUDE_DIR	        C:/gstreamer/1.0/msvc_x86_64/include/gstreamer-1.0
GSTREAMER_gstreamer_LIBRARY	      C:/gstreamer/1.0/msvc_x86_64/lib/gstreamer-1.0.lib
GSTREAMER_pbutils_LIBRARY	        C:/gstreamer/1.0/msvc_x86_64/lib/gstpbutils-1.0.lib
GSTREAMER_riff_LIBRARY	          C:/gstreamer/1.0/msvc_x86_64/lib/gstriff-1.0.lib
GSTREAMER_video_LIBRARY	          C:/gstreamer/1.0/msvc_x86_64/lib/gstvideo-1.0.lib

BUILD_opencv_java: OFF
BUILD_opencv_python3: OFF
BUILD_PERF_TESTS: OFF
BUILD_TESTS: OFF
BUILD_opencv_cudaarithm: OFF
BUILD_opencv_cudafilters: OFF
BUILD_opencv_cudafeatures2d: OFF
BUILD_opencv_cudaimgproc: OFF
BUILD_opencv_cudacodec: OFF
BUILD_opencv_cudaobjdetect: OFF
BUILD_opencv_cudaoptflow: OFF
BUILD_opencv_cudalegacy: OFF
BUILD_opencv_stitching: OFF
BUILD_opencv_videostab: OFF
BUILD_opencv_superres: OFF
BUILD_opencv_xfeatures2d: OFF
BUILD_opencv_xphoto: OFF
BUILD_opencv_photo: OFF
BUILD_opencv_face: OFF
BUILD_opencv_gapi: OFF
INSTALL_C_EXAMPLES: OFF
INSTALL_PYTHON_EXAMPLES: OFF
BUILD_EXAMPLES: OFF
BUILD_JAVA: OFF
WITH_ADE: OFF


BUILD_opencv_world: IF ON THIS WILL CREATE A SINGLE FILE FOR YOU TO LINK, IT IS QUITE CONVENIENT BUT FOR ME IT WAS IMPOSSIBLE TO GET IT WORKING.

CUDA_ARCH_BIN: SELECT ONLY ONE DEPENDING ON YOUR GPU-CUDA ARQUITECTURE (this info can be find online for your GPU) => mine was 8.6
CUDA_CUDART_LIBRARY: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/cudart.lib
CUDA_CUDA_LIBRARY: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/cuda.lib
CUDA_NVCC_EXECUTABLE: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe
CUDA_OpenCL_LIBRARY: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64/OpenCL.lib

CHECK THAT ALL CUDA MODULES HAVE THEIR path correctly, except for nvcuvid which is not important, for me the one that did not find was:
CUDA_nvToolsExt_LIBRARY: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64/nvToolsExt64_1.lib (Is here because i manually move it to CUDA dir, It did not find it because of the name, normally will search for nvToolsExt.lib)

```
If everything is correct you can click one last time on configure, then generate (if you still have errors, you probably type one path wrong, solve it and configure again).

Finally open a CMD as superuser, go to the build folder (for me `C:\opencv_cuda\build`) and type `OpenCV.sln` and click ENTER. This will open Visual Studio.
On visual studio change from **Debug to Releasee**, then click on CMakeTragets, will open the targets, then right click on **BUIL_ALL** and click **build**.
This will take a will, in the debug window you can see the progress, when finished you must checked the succes of the compilation (you may have forgot to take one module out and the compilation may have 1 or 2 errors, this is not important while the correct modules are the ones you need).
Finally right click on the target **INSTALL** and click build, this will take very little time.

If averything works as expected you will have your own custom compilation of OpenCV in `C:\opencv_cuda\build\install`, you can copy its contents and paste them into it's own carpet, for example: `C:\opencv`.

Congratulations you made it !!!

### CMake configuration

If everything else worked okay this will be the easy part, you just have to modify a few paths in the CMakeLists.txt of this folder with your own. They are the following, by this point you must know exactly where they need to point :).

```cmake
set(CMAKE_PREFIX_PATH 
    "C:/opencv_cuda"
    "C:/libtorch/libtorch"
)
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
set(OpenCV_DIR "C:/opencv_cuda")
find_library(NVTOOLSEXT_LIB 
    NAMES nvToolsExt nvToolsExt64_1
    PATHS
        "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"
        "C:/Program Files/NVIDIA Corporation/nvToolsExt/lib/x64"
)
link_directories("C:/opencv_cuda/x64/vc17/staticlib")
```

Then try to build it, it will take some time and it may print a lot of warnings with complicated and big text, but if the build return 0 you can easily run the application with:

```bash
.\obs_avoid_full_cpp_mt.exe -c -y
```


If somthing does not compile you can uncomment the following lines of code and comment the main programm, to check with the most simple programm if libtorch and opencv are detecting cuda:

```cmake
# =======================================MAIN========================================
add_executable(obs_avoid_full_cpp_mt
    src/main.cpp
    src/yolov11.cpp
    src/helper.cpp
    src/globals.cpp
)
target_include_directories(obs_avoid_full_cpp_mt PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(obs_avoid_full_cpp_mt PRIVATE ${TORCH_INCLUDE_DIRS})
target_compile_definitions(obs_avoid_full_cpp_mt PRIVATE
    -D_WIN32
    -D_WINDOWS
    -DOPENCV_STATIC
    -DNOMINMAX
    -D_USE_MATH_DEFINES
)
target_link_libraries(obs_avoid_full_cpp_mt ${TORCH_LIBRARIES} ${OpenCV_LIBS} ws2_32)
# ===================================================================================

# =========UNCOMENT FOR CUDA CHECKING============
# add_executable(cuda_check src/test_win.cpp)

# target_include_directories(cuda_check PRIVATE ${OpenCV_INCLUDE_DIRS})
# target_include_directories(cuda_check PRIVATE ${TORCH_INCLUDE_DIRS})

# # Definiciones para compilación estática
# target_compile_definitions(cuda_check PRIVATE
#     -D_WIN32
#     -D_WINDOWS
#     -DOPENCV_STATIC
#     -DNOMINMAX
#     -D_USE_MATH_DEFINES
# )

# target_link_libraries(cuda_check
#     ${OpenCV_LIBS}
#     ${TORCH_LIBRARIES}
# )
# if (TARGET CUDA::nvToolsExt)
#     target_link_libraries(cuda_check CUDA::nvToolsExt)
# endif()
# ===============================================

```
