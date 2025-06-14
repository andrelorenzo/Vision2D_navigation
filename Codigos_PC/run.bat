@echo off
REM ==== Ruta a Visual Studio 2022 Community - AJUSTAR si es necesario ====
CALL "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

REM ==== Configurar CudaToolkitDir (necesario para MSBuild) ====
set CudaToolkitDir=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9

REM ==== Ir al directorio del proyecto ====
cd /d %~dp0

REM ==== Ejecutar CMake ====
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 ^
 -DCMAKE_PREFIX_PATH="C:/opencv/build/x64/vc16/lib;C:/libtorch/libtorch"

REM ==== Compilar el proyecto ====
cmake --build build --config Release
