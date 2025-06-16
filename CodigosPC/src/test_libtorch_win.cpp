#include <torch/torch.h>
#include <iostream>

int main() {
    if (torch::cuda::is_available()) {
        std::cout << "CUDA está disponible. Usando GPU." << std::endl;
    } else {
        std::cout << "CUDA no está disponible. Usando CPU." << std::endl;
    }

    torch::Tensor t = torch::rand({3, 3}).to(torch::kCUDA);
    std::cout << "Tensor:\n" << t << std::endl;

    return 0;
}