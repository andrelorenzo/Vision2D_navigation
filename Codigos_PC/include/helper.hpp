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
void draw_mean_slope_arrow_sobel(cv::Mat& vis, const cv::Mat& depth_map);
void annotate_with_depth(cv::Mat& frame, const cv::Mat& depth_map, std::vector<ObjectBBox>& detections);
cv::Mat estimate_depth_map(torch::jit::script::Module& model, const cv::Mat& frame,
                           const cv::Size& input_size,
                           const std::vector<float>& mean,
                           const std::vector<float>& std,
                           bool swapRB, bool crop);
// wrappers de modelos de estimacion de profundidad
cv::Mat estimate_midas_depth_v21(torch::jit::script::Module& model, const cv::Mat& frame);
cv::Mat estimate_midas_dpt_hybrid_384(torch::jit::script::Module& model, const cv::Mat& frame) ;
cv::Mat estimate_depth_anything_v2_outdoor(torch::jit::script::Module& model, const cv::Mat& frame);
cv::Mat estimate_depth_pro_bnb4(torch::jit::script::Module& model, const cv::Mat& frame);
