#pragma once

#include "main.hpp"

// camera
bool recvAll(int sock, uint8_t* buffer, size_t length);
cv::Mat get_frame_from_tcp(int sock);
cv::Mat get_frame_from_camera(cv::VideoCapture& cap);

// opencv depth and gradient calculations
cv::Vec2f draw_mean_slope_arrow_sobel(cv::Mat& vis, const cv::Mat& depth_map_input);
cv::Mat exponential_smoothing(const cv::Mat current, cv::Mat future, float alpha);
cv::Mat normalize_depth_with_percentile(const cv::Mat& input, float lower_percent = 1.0f, float upper_percent = 99.0f);
cv::Mat estimate_depth_map(torch::jit::script::Module& model, const cv::Mat& frame,
                           const cv::Size& input_size,
                           const std::vector<float>& mean,
                           const std::vector<float>& std,
                           bool swapRB, bool crop);
// wrappers de modelos de estimacion de profundidad
cv::Mat estimate_depth_variant(DepthModel& model, const cv::Mat& frame);

// extrapolation of distance using yolov11n
void annotate_with_depth(cv::Mat& frame,
                         std::vector<ObjectBBox>& detections,
                         const std::unordered_map<std::string, std::pair<float, float>>& object_sizes);
                         
std::unordered_map<std::string, std::pair<float, float>> load_object_sizes(const std::string& filename);
bool calibrate_and_scale_midas(cv::Mat& depth_midas_normalized,
                               const std::vector<ObjectBBox>& detections,
                               const std::unordered_map<std::string, std::pair<float, float>>& object_sizes,
                               cv::Mat& depth_scaled_out);
float estimate_distance_from_yolo(const ObjectBBox& det,const std::unordered_map<std::string, std::pair<float, float>>& sizes);
