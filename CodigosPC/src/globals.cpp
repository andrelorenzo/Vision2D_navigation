#include "main.hpp"

cv::Mat depth_scaled_for_debug;
bool do_bench = true;

cv::Mat shared_frame;
std::vector<ObjectBBox> current_detections;
cv::Mat current_depth;

std::atomic<bool> stop_flag(false);
std::atomic<bool> midas_ready(false);
std::atomic<bool> yolo_ready(false);

std::mutex frame_mutex;
std::mutex yolo_mutex;
std::mutex depth_mutex;
