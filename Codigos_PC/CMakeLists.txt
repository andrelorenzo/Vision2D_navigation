cmake_minimum_required(VERSION 3.22.1)
project(obstacle_avoidance LANGUAGES CXX CUDA)
cmake_policy(SET CMP0146 OLD)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wno-dev")

# Ruta a cuDNN detectada manualmente
set(CUDNN_LIBRARY "/usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn.so")
set(CUDNN_INCLUDE_DIR "/usr/local/cuda-12.1/targets/x86_64-linux/include")
set(CMAKE_PREFIX_PATH "/home/andrelorent/thirds/libtorch")

# include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# === OpenCV ===
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(/usr/local/cuda-12.1/targets/x86_64-linux/lib)
include_directories(/usr/local/cuda-12.1/targets/x86_64-linux/include)
find_package(OpenCV REQUIRED)

# === libtorch ===
find_package(Torch REQUIRED)


# Full code with both models in cpp and multitreding
add_executable(obs_avoid_full_cpp_mt
    src/main.cpp
    # src/serialib.cpp
    src/yolov11.cpp
    src/helper.cpp
    src/globals.cpp
)
target_link_libraries(obs_avoid_full_cpp_mt PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(obs_avoid_full_cpp_mt PRIVATE ${OpenCV_INCLUDE_DIRS})

#Full code with both models in cpp 
add_executable(obs_avoid_full_cpp
    old/main_fullcpp.cpp
    old/serialib.cpp
    src/yolov11.cpp
)
target_link_libraries(obs_avoid_full_cpp PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(obs_avoid_full_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})


# #Test of ESPNOW bridge + Serial 
# add_executable(obstacle_avoidance 
#     src/main.cpp
#     src/serialib.cpp
# )
# target_link_libraries(obstacle_avoidance PRIVATE ${OpenCV_LIBS})
# target_include_directories(obstacle_avoidance PRIVATE ${OpenCV_INCLUDE_DIRS})

# #Test of WIfi + YOLOv11
# add_executable(obstacle_avoidance_wifi 
#     src/main_wifi.cpp
#     src/serialib.cpp
#     src/yolov11.cpp
# )
# target_link_libraries(obstacle_avoidance_wifi PRIVATE ${OpenCV_LIBS})
# target_include_directories(obstacle_avoidance_wifi PRIVATE ${OpenCV_INCLUDE_DIRS})

# # Get camera params
# add_executable(get_camera_params
#     src/main_get_camera_params.cpp
# )
# target_link_libraries(get_camera_params PRIVATE ${OpenCV_LIBS})
# target_include_directories(get_camera_params PRIVATE ${OpenCV_INCLUDE_DIRS})


# set(ORT_PATH /home/andrelorent/programming/Vision2D_navigation/Codigos_PC/libs/onnxruntime-linux-x64-gpu-1.22.0)

# add_executable(depth_anything_app src/main_onnx_rt.cpp)

# target_include_directories(depth_anything_app PRIVATE ${ORT_PATH}/include)
# target_link_directories(depth_anything_app PRIVATE ${ORT_PATH}/lib)

# target_link_libraries(depth_anything_app
#     PRIVATE
#     ${OpenCV_LIBS}
#     onnxruntime
#     onnxruntime_providers_cuda
# )
