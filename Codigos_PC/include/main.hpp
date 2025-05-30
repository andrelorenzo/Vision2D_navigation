#pragma once

#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "yolov11.hpp"

#define SERVER_IP   "192.168.4.1"
#define SERVER_PORT 8888
#define CAPTURE_CMD "CAPT"

#define DEV_TRHS 1
#define MIN_TRHS 1

// frame mutex
extern std::mutex frame_mutex;
extern std::condition_variable frame_cv;
extern cv::Mat shared_frame;

// camera buffer
extern cv::Mat buffer_depth[2];
extern cv::Mat buffer_yolo[2];
extern std::atomic<int> frame_index;
extern bool buffer_full;


extern std::atomic<bool> new_frame_ready, stop_flag;
extern cv::Mat current_depth;
extern std::vector<ObjectBBox> current_detections;
extern std::atomic<int> threads_done;
extern std::mutex threads_mutex;
extern std::condition_variable threads_cv;