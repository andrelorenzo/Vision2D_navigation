cmake_minimum_required(VERSION 3.23)
project(obstacle_avoidance)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OpenCV_DIR "C:/opencv411/install")  # <-- ajusta según la ubicación real

set(CMAKE_BUILD_TYPE Debug)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(obstacle_avoidance 
    src/main.cpp
    src/serialib.cpp
)

add_executable(obstacle_avoidance_wifi 
    src/main_wifi.cpp
    src/serialib.cpp
)
add_executable(obstacle_avoidance_wifi_midas 
    src/main_wifi_midas.cpp
    src/serialib.cpp
)

target_link_libraries(obstacle_avoidance PRIVATE ${OpenCV_LIBS})
target_link_libraries(obstacle_avoidance_wifi PRIVATE ${OpenCV_LIBS} ws2_32)
target_link_libraries(obstacle_avoidance_wifi_midas PRIVATE ${OpenCV_LIBS} ws2_32)
# # Ruta donde están las DLLs de OpenCV compiladas con MinGW
# set(OpenCV_DLL_DIR "C:/opencv411/install/x64/mingw/bin")

# # Lista de DLLs necesarias (puedes añadir más si usas otros módulos)
# set(OpenCV_DLLS
#     opencv_core411.dll
#     opencv_imgproc411.dll
#     opencv_highgui411.dll
#     opencv_imgcodecs411.dll
# )

# # Copiar las DLLs al directorio de salida (build/)
# foreach(DLL ${OpenCV_DLLS})
#     add_custom_command(TARGET obstacle_avoidance POST_BUILD
#         COMMAND ${CMAKE_COMMAND} -E copy_if_different
#         "${OpenCV_DLL_DIR}/${DLL}"
#         $<TARGET_FILE_DIR:obstacle_avoidance>
#     )
# endforeach()