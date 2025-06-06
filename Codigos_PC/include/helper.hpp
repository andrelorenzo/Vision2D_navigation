#pragma once

#include "main.hpp"


// treahds
void signal_thread_done();
void wait_for_threads(int required);

// camera
bool recvAll(int sock, uint8_t* buffer, size_t length);
cv::Mat get_frame_from_tcp(int sock);
cv::Mat get_frame_from_camera(cv::VideoCapture& cap);

// opencv depth and gradient calculations
void draw_mean_slope_arrow_sobel(cv::Mat& vis, const cv::Mat& depth_map_input) ;
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
                         cv::Mat& depth_map,
                         std::vector<ObjectBBox>& detections,
                         const std::unordered_map<std::string, std::pair<float, float>>& object_sizes,
                         bool& depth_scaled_flag);
std::unordered_map<std::string, std::pair<float, float>> load_object_sizes(const std::string& filename);
void scale_depth_map(cv::Mat& depth_map, float depth_reference_m, float midas_value_reference) ;
float estimate_distance_from_yolo(const ObjectBBox& det,
                                  const std::unordered_map<std::string, std::pair<float, float>>& sizes,
                                  float focal_px);
