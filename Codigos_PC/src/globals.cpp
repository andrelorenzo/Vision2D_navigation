#include "main.hpp"

std::mutex frame_mutex;
std::condition_variable frame_cv;
cv::Mat shared_frame;

// camera buffer
cv::Mat buffer_depth[2];
cv::Mat buffer_yolo[2];
std::atomic<int> frame_index = 0;
bool buffer_full = false;

std::atomic<bool> new_frame_ready(false);
std::atomic<bool> stop_flag(false);
cv::Mat current_depth;
std::vector<ObjectBBox> current_detections;
std::atomic<int> threads_done{0};
std::mutex threads_mutex;
std::condition_variable threads_cv;
