cmake_minimum_required(VERSION 3.22.1)
project(obstacle_avoidance)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# === OpenCV ===
set(BUILD_SHARED_LIBS OFF)
add_definitions(-DOPENCV_STATIC)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# === libtorch ===
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libs/libtorch")
find_package(Torch REQUIRED)

# Full code with both models in cpp 
add_executable(obs_avoid_full_cpp
    src/main_fullcpp.cpp
    src/serialib.cpp
    src/yolov11.cpp
)
target_link_libraries(obs_avoid_full_cpp PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(obs_avoid_full_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})

# Full code with both models in cpp and multitreding
add_executable(obs_avoid_full_cpp_mt
    src/main_fullcpp_mt.cpp
    src/serialib.cpp
    src/yolov11.cpp
    src/aux.cpp
    src/globals.cpp
)
target_link_libraries(obs_avoid_full_cpp_mt PRIVATE ${TORCH_LIBRARIES} ${OpenCV_LIBS})
target_include_directories(obs_avoid_full_cpp_mt PRIVATE ${OpenCV_INCLUDE_DIRS})

# Test of ESPNOW bridge + Serial 
add_executable(obstacle_avoidance 
    src/main.cpp
    src/serialib.cpp
)
target_link_libraries(obstacle_avoidance PRIVATE ${OpenCV_LIBS})
target_include_directories(obstacle_avoidance PRIVATE ${OpenCV_INCLUDE_DIRS})

# Test of WIfi + YOLOv11
add_executable(obstacle_avoidance_wifi 
    src/main_wifi.cpp
    src/serialib.cpp
    src/yolov11.cpp
)
target_link_libraries(obstacle_avoidance_wifi PRIVATE ${OpenCV_LIBS})
target_include_directories(obstacle_avoidance_wifi PRIVATE ${OpenCV_INCLUDE_DIRS})

# Get camera params
# add_executable(get_camera_params
#     src/main_get_camera_params.cpp
# )
# target_link_libraries(get_camera_params PRIVATE ${OpenCV_LIBS})
# target_include_directories(get_camera_params PRIVATE ${OpenCV_INCLUDE_DIRS})
