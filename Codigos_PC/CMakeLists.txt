cmake_minimum_required(VERSION 3.22.1)
project(obstacle_avoidance)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wno-dev")
# Ruta a cuDNN detectada manualmente
set(CUDNN_LIBRARY "/usr/local/cuda-12.1/targets/x86_64-linux/lib/libcudnn.so")
set(CUDNN_INCLUDE_DIR "/usr/local/cuda-12.1/targets/x86_64-linux/include")
set(USE_CUDNN ON)
# include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# === OpenCV ===
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(/usr/local/cuda-12.1/targets/x86_64-linux/lib)
include_directories(/usr/local/cuda-12.1/targets/x86_64-linux/include)

# === libtorch ===

option(USE_CUDA "Use LibTorch with CUDA" ON)  # Valor por defecto ON

if (USE_CUDA)
    message(STATUS "Compilando con soporte CUDA")
    set(CMAKE_PREFIX_PATH "/home/andrelorent/libtorch")
else()
    message(STATUS "Compilando con LibTorch CPU")
    set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libs/libtorch")
endif()

find_package(Torch REQUIRED)


# Full code with both models in cpp and multitreding
add_executable(obs_avoid_full_cpp_mt
    src/main_fullcpp_mt.cpp
    src/serialib.cpp
    src/yolov11.cpp
    src/helper.cpp
    src/globals.cpp
)
target_link_libraries(obs_avoid_full_cpp_mt PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(obs_avoid_full_cpp_mt PRIVATE ${OpenCV_INCLUDE_DIRS})

#Full code with both models in cpp 
add_executable(obs_avoid_full_cpp
    src/main_fullcpp.cpp
    src/serialib.cpp
    src/yolov11.cpp
)
target_link_libraries(obs_avoid_full_cpp PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(obs_avoid_full_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})




# Test of ESPNOW bridge + Serial 
# add_executable(obstacle_avoidance 
#     src/main.cpp
#     src/serialib.cpp
# )
# target_link_libraries(obstacle_avoidance PRIVATE ${OpenCV_LIBS})
# target_include_directories(obstacle_avoidance PRIVATE ${OpenCV_INCLUDE_DIRS})

# Test of WIfi + YOLOv11
# add_executable(obstacle_avoidance_wifi 
#     src/main_wifi.cpp
#     src/serialib.cpp
#     src/yolov11.cpp
# )
# target_link_libraries(obstacle_avoidance_wifi PRIVATE ${OpenCV_LIBS})
# target_include_directories(obstacle_avoidance_wifi PRIVATE ${OpenCV_INCLUDE_DIRS})

# Get camera params
# add_executable(get_camera_params
#     src/main_get_camera_params.cpp
# )
# target_link_libraries(get_camera_params PRIVATE ${OpenCV_LIBS})
# target_include_directories(get_camera_params PRIVATE ${OpenCV_INCLUDE_DIRS})
