#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

int main() {
    // Verificar CUDA en LibTorch
    if (torch::cuda::is_available()) {
        std::cout << "CUDA está disponible en LibTorch." << std::endl;
        std::cout << "Número de dispositivos CUDA: " << torch::cuda::device_count() << std::endl;
    } else {
        std::cout << "CUDA NO está disponible en LibTorch." << std::endl;
    }

    // Verificar CUDA en OpenCV
    int cuda_device_count = cv::cuda::getCudaEnabledDeviceCount();
    if (cuda_device_count > 0) {
        std::cout << "CUDA está disponible en OpenCV. Dispositivos detectados: " << cuda_device_count << std::endl;
        cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    } else {
        std::cout << "CUDA NO está disponible en OpenCV." << std::endl;
    }

    return 0;
}
