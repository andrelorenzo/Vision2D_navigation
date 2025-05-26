cmake_minimum_required(VERSION 3.23)
project(obstacle_avoidance)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# === OpenCV ===
set(OpenCV_DIR "C:/opencv/build") 
set(BUILD_SHARED_LIBS OFF)
add_definitions(-DOPENCV_STATIC)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# === libtorch ===
find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})

add_executable(obstacle_avoidance 
    src/main.cpp
    src/serialib.cpp
)
target_link_libraries(obstacle_avoidance PRIVATE ${OpenCV_LIBS})

add_executable(obstacle_avoidance_wifi 
    src/main_wifi.cpp
    src/serialib.cpp
)
target_link_libraries(obstacle_avoidance_wifi PRIVATE ${OpenCV_LIBS} ws2_32)

add_executable(obstacle_avoidance_wifi_midas 
    src/detect_libtorch.cpp
    src/main_wifi_midas.cpp
    src/serialib.cpp
)
target_link_libraries(obstacle_avoidance_wifi_midas PRIVATE "${TORCH_LIBRARIES}" ${OpenCV_LIBS} ws2_32)
target_include_directories(obstacle_avoidance_wifi_midas PRIVATE ${OpenCV_INCLUDE_DIRS})

# Recomendado: asegurar que MSVC encuentre las DLL de libtorch en tiempo de ejecución

set_property(TARGET obstacle_avoidance_wifi_midas PROPERTY VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${TORCH_INSTALL_PREFIX}/lib")
