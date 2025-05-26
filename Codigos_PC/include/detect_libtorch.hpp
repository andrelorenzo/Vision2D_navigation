#pragma once

#include <torch/script.h>
// #include <torch/csrc/api/include/torch/torch.h>
#include <ATen/ops/upsample_bilinear2d.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>


struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
    float depth;
};

// Preprocesa una imagen de entrada para el modelo (normaliza y ajusta tamaño)
torch::Tensor preprocess_image(const cv::Mat& img, int height, int width);

// Estima la profundidad promedio dentro de una caja delimitadora
float get_depth_in_bbox(const cv::Mat& depth_map, const cv::Rect& box);

// Ejecuta inferencia con YOLO y MiDaS usando LibTorch y retorna las detecciones
void detect_with_depth(const cv::Mat& image,
                       torch::jit::script::Module& yolo_model,
                       torch::jit::script::Module& midas_model,
                       std::vector<Detection>& output,
                       const std::vector<std::string>& class_list);

