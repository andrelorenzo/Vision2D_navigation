cmake_minimum_required(VERSION 3.22.1)
project(obstacle_avoidance LANGUAGES CXX)

# CRT estática en Release (equivalente a /MT)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "" FORCE)

cmake_policy(SET CMP0146 OLD)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Forzar uso de CRT estática (/MT) en Release y Debug
foreach (flag_var
  CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_DEBUG
  CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
  string(REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
endforeach()

# Ruta a dependencias
set(CMAKE_PREFIX_PATH 
    "C:/opencv_cuda"
    "C:/libtorch/libtorch"
)

# CUDA Toolkit
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE INTERNAL "")
set(CudaToolkitDir "${CUDA_TOOLKIT_ROOT_DIR}")

# Incluir cabeceras propias
include_directories(${PROJECT_SOURCE_DIR}/include)

# OpenCV (estático)
set(OpenCV_STATIC ON)
set(OpenCV_DIR "C:/opencv_cuda")
find_package(OpenCV REQUIRED)

# Torch
find_package(Torch REQUIRED)

# Workaround: crear CUDA::nvToolsExt si no existe
find_library(NVTOOLSEXT_LIB 
    NAMES nvToolsExt nvToolsExt64_1
    PATHS
        "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64"
        "C:/Program Files/NVIDIA Corporation/nvToolsExt/lib/x64"
)

if (NVTOOLSEXT_LIB)
    message(STATUS "nvToolsExt encontrado en: ${NVTOOLSEXT_LIB}")
    
    if (NOT TARGET CUDA::nvToolsExt)
        add_library(CUDA::nvToolsExt UNKNOWN IMPORTED GLOBAL)
        set_target_properties(CUDA::nvToolsExt PROPERTIES
            IMPORTED_LOCATION "${NVTOOLSEXT_LIB}"
        )
    else()
        message(STATUS "CUDA::nvToolsExt ya estaba definido por otro paquete.")
    endif()
else()
    message(WARNING "nvToolsExt.lib no encontrado. CUDA::nvToolsExt no será enlazado.")
endif()


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


## =========UNCOMENT FOR CUDA CHECKING============
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
## ===============================================




if (TARGET CUDA::nvToolsExt)
    target_link_libraries(obs_avoid_full_cpp_mt CUDA::nvToolsExt)
endif()

link_directories("C:/opencv_cuda/x64/vc17/staticlib")
