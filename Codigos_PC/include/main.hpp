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
#include <variant>
#include <functional>
#include <unordered_map>
#include <sstream>
#include <winsock2.h>
#include <ws2tcpip.h>
#define CLOSESOCKET closesocket
#pragma comment(lib, "ws2_32.lib")

#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "yolov11.hpp"
#include "profiler.hpp"

#define SERVER_IP   "192.168.4.1"
#define SERVER_PORT 8888
#define CAPTURE_CMD "CAPT"

#define DEV_TRHS 1
#define MIN_TRHS 1

/*Expressed in pixels, conversion to mm in x: 0.00421875, in y: 0.004375*/
#define FOCALX 1472.768 / 2.0
#define FOCALY 1449.567 / 2.0
#define CENTERX 86.747
#define CENTERY 50.412
#define MAXPOLY 20

// debug varibles
extern cv::Mat depth_scaled_for_debug;                // for mouse callback [DEBUG]
extern bool do_bench;

// shared objects between all flows
extern cv::Mat shared_frame;                          // shared frame between yolo, midas and capture
extern std::vector<ObjectBBox> current_detections;    // output shared object between yolo and main
extern cv::Mat current_depth;                         // Output shared frame between midas and main

// flow control for new frame
extern std::atomic<bool> stop_flag;                   // atomic stop flag to exit all threads
extern std::atomic<bool> midas_ready;                 // atomic new frame flag for midas
extern std::atomic<bool> yolo_ready;                  // atomic new frame flag for yolo

// mutex for each thread
extern std::mutex frame_mutex;                        // mutex for camera
extern std::mutex depth_mutex;                        // mutex for midas
extern std::mutex yolo_mutex;                         // mutex for yolo


// Quality of Life and templates
using DepthModel = std::variant<cv::dnn::Net, torch::jit::script::Module>;
using DepthEstimationFn = std::function<cv::Mat(DepthModel&, const cv::Mat&)>;

template<class... Ts>
struct overloaded : Ts... { using Ts::operator()...; };
template<class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;