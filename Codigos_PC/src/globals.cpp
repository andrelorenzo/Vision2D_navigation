#include "main.hpp"


std::atomic<bool> stop_flag(false);
std::mutex depth_mutex;
std::mutex yolo_mutex;
cv::Mat current_depth;
std::vector<ObjectBBox> current_detections;
std::mutex threads_mutex;
std::condition_variable threads_cv;
std::atomic<uint64_t> threads_done_indexed{0};
std::atomic<uint64_t> depth_done_frame_index{0};
std::atomic<uint64_t> yolo_done_frame_index{0};

cv::Mat depth_scaled_for_debug;
cv::Mat shared_frame;
std::mutex frame_mutex;
std::atomic<uint64_t> global_frame_index{0};
