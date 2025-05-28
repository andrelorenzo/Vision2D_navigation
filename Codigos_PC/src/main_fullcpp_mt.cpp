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

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "yolov11.hpp"


#include <opencv2/opencv.hpp>
#include <vector>

#define SERVER_IP   "192.168.4.1"
#define SERVER_PORT 8888
#define CAPTURE_CMD "CAPT"

#define DEV_TRHS 4
#define MIN_TRHS 10
std::mutex frame_mutex;
std::condition_variable frame_cv;
cv::Mat shared_frame;


std::mutex benchmark_mutex;
double midas_time_total = 0.0;
double yolo_time_total = 0.0;
int benchmark_count = 0;


std::atomic<bool> new_frame_ready(false);
std::atomic<bool> stop_flag(false);

cv::Mat current_depth;
std::vector<ObjectBBox> current_detections;

std::atomic<int> threads_done{0};
std::chrono::high_resolution_clock::time_point frame_start_time;



bool recvAll(int sock, uint8_t* buffer, size_t length) {
    size_t total = 0;
    while (total < length) {
        ssize_t bytes = recv(sock, buffer + total, length - total, 0);
        if (bytes <= 0) return false;
        total += bytes;
    }
    return true;
}
cv::Mat get_frame_from_tcp(int sock) {
    send(sock, CAPTURE_CMD, 4, 0);

    uint8_t sizeBytes[4];
    if (!recvAll(sock, sizeBytes, 4)) return cv::Mat();

    int imgSize = (sizeBytes[0] << 24) | (sizeBytes[1] << 16) | (sizeBytes[2] << 8) | sizeBytes[3];
    if (imgSize <= 0 || imgSize > 150000) return cv::Mat();

    std::vector<uint8_t> imgBuffer(imgSize);
    if (!recvAll(sock, imgBuffer.data(), imgSize)) return cv::Mat();

    return cv::imdecode(imgBuffer, cv::IMREAD_COLOR);
}

cv::Mat get_frame_from_camera(cv::VideoCapture& cap) {
    cv::Mat frame;
    cap >> frame;
    return frame;
}


void draw_mean_slope_arrow_sobel(cv::Mat& vis, const cv::Mat& depth_map) {
    if (depth_map.empty() || depth_map.type() != CV_32F) return;

    cv::Mat dx, dy;
    cv::Sobel(depth_map, dx, CV_32F, 1, 0, 3);  // Derivada en X
    cv::Sobel(depth_map, dy, CV_32F, 0, 1, 3);  // Derivada en Y

    // Invertimos los gradientes porque buscamos la bajada
    dx = -dx;
    dy = -dy;

    // Cálculo de dirección promedio
    cv::Scalar mean_dx = cv::mean(dx);
    cv::Scalar mean_dy = cv::mean(dy);
    cv::Mat grad_mag;
    cv::magnitude(dx, dy, grad_mag);

    cv::Scalar stddev;
    cv::meanStdDev(grad_mag, cv::noArray(), stddev);

    cv::Point2f dir(mean_dx[0], mean_dy[0]);
    float norm = std::sqrt(dir.x * dir.x + dir.y * dir.y);

    if (stddev[0] < DEV_TRHS || norm < MIN_TRHS ){
        norm = 0.0f;
        dir = cv::Point2f(0.f, 0.f);
    }

    // Escalado proporcional (opcional: ajustar factor_visual para visibilidad)
    const float factor_visual = 10.0f; // Aumentar si los vectores son muy pequeños
    dir *= factor_visual;

    // Dibujar si procede
    if (norm > 0.0f) {
        cv::Point2f center(depth_map.cols / 2.0f, depth_map.rows / 2.0f);
        cv::Point2f end = center + dir;
        cv::arrowedLine(vis, center, end, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }

    // Mostrar magnitud
    std::ostringstream text;
    text << "Mag: " << std::fixed << std::setprecision(2) << norm;
    cv::putText(vis, text.str(), cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
}




void frame_capture_thread(bool use_tcp, int sock, cv::VideoCapture& cap) {
    while (!stop_flag) {
        cv::Mat frame = use_tcp ? get_frame_from_tcp(sock) : get_frame_from_camera(cap);
        if (frame.empty()) continue;

        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            shared_frame = frame.clone();
            new_frame_ready = true;
            threads_done = 0;
            frame_start_time = std::chrono::high_resolution_clock::now(); // ⬅ iniciar cronómetro
        }
        frame_cv.notify_all();
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

cv::Mat estimate_depth_map(cv::dnn::Net& midas, const cv::Mat& frame) {
    cv::Mat resized, input_blob, depth_map;

    cv::resize(frame, resized, cv::Size(256, 256));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    input_blob = cv::dnn::blobFromImage(resized);
    midas.setInput(input_blob);
    depth_map = midas.forward();
    depth_map = depth_map.reshape(1, 256);
    cv::resize(depth_map, depth_map, frame.size());

    return depth_map;
}
void annotate_with_depth(cv::Mat& frame, const cv::Mat& depth_map, std::vector<ObjectBBox>& detections) {
    for (auto& bbox : detections) {
        int x1 = std::clamp(static_cast<int>(bbox.x1), 0, frame.cols - 1);
        int y1 = std::clamp(static_cast<int>(bbox.y1), 0, frame.rows - 1);
        int x2 = std::clamp(static_cast<int>(bbox.x2), 0, frame.cols - 1);
        int y2 = std::clamp(static_cast<int>(bbox.y2), 0, frame.rows - 1);

        // Asegurarse de que x2 > x1 y y2 > y1
        if (x2 <= x1 || y2 <= y1) continue;

        cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
        cv::Mat depth_roi = depth_map(roi);
        cv::resize(depth_map, depth_map, frame.size());
        CV_Assert(depth_map.size() == frame.size());
        float distance = static_cast<float>(cv::mean(depth_roi)[0]);

        std::ostringstream label;
        label << bbox.label << " " << std::fixed << std::setprecision(2)
              << bbox.conf << " D:" << distance;

        bbox.draw(frame);
        cv::putText(frame, label.str(), cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 2);
    }
}
void midas_thread(cv::dnn::Net& midas_net) {
    while (!stop_flag) {
        std::unique_lock<std::mutex> lock(frame_mutex);
        frame_cv.wait(lock, [] { return new_frame_ready || stop_flag; });
        if (stop_flag) break;
        cv::Mat frame = shared_frame.clone();
        lock.unlock();

        auto t0 = std::chrono::high_resolution_clock::now();
        current_depth = estimate_depth_map(midas_net, frame);
        auto t1 = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::lock_guard<std::mutex> bench_lock(benchmark_mutex);
        midas_time_total += duration;

        threads_done++;
    }
}


void yolo_thread(YOLOv11& yolo_model) {
    while (!stop_flag) {
        std::unique_lock<std::mutex> lock(frame_mutex);
        frame_cv.wait(lock, [] { return new_frame_ready || stop_flag; });
        if (stop_flag) break;
        cv::Mat frame = shared_frame.clone();
        lock.unlock();

        auto t0 = std::chrono::high_resolution_clock::now();
        current_detections = yolo_model.detect(frame);
        auto t1 = std::chrono::high_resolution_clock::now();

        double duration = std::chrono::duration<double, std::milli>(t1 - t0).count();
        std::lock_guard<std::mutex> bench_lock(benchmark_mutex);
        yolo_time_total += duration;

        threads_done++;
    }
}int main(int argc, char** argv) {
    bool use_tcp = true;
    bool use_yolo = false;
    bool use_bench_marking = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == " --camera" || arg == "-c") use_tcp = false;
        if (arg == " --yolo" || arg == "-y") use_yolo = true;
        if (arg == "--nobench" || arg == "-nb") use_bench_marking = false;
        if (arg == "--help" || arg == "-h"){
          std::cout << "Usage: ./obs_avoid_full_cpp_mt [PARAMS]" << std::endl;
          std::cout << "\t -c, --camera\t If selected PC camera will be selected" << std::endl;
          std::cout << "\t -c, --yolo\t If selected YOLOv11n will be run in parallel" << std::endl;
          std::cout << "\t -nb, --nobench\t If selected no benchmarking will be printed" << std::endl;
          return 0;
        }
    }

    cv::dnn::Net midas = cv::dnn::readNet("../models/model-small.onnx");
    midas.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    midas.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    YOLOv11 yolo_model("../models/yolo11n.onnx", 0.45f, 0.45f, [](int id, const std::string&) {
        return id >= 0 && id <= 16;
    });

    int sock = -1;
    cv::VideoCapture cap;
    if (use_tcp) {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in server{};
        server.sin_family = AF_INET;
        server.sin_port = htons(SERVER_PORT);
        inet_pton(AF_INET, SERVER_IP, &server.sin_addr);
        while (connect(sock, (sockaddr*)&server, sizeof(server)) < 0)
            std::cout << "\033[2J\033[1;1H";
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            std::cout << "Connecting to camera..." << std::endl;

    } else {
        cap.open(0);
    }

    std::thread t_capture(frame_capture_thread, use_tcp, sock, std::ref(cap));
    std::thread t_midas(midas_thread, std::ref(midas));
    std::thread t_yolo;
    if (use_yolo)
        t_yolo = std::thread(yolo_thread, std::ref(yolo_model));

    int frame_count = 0;
    double total_time = 0.0;

    while (true) {
        while (threads_done.load() < (use_yolo ? 2 : 1))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lock(frame_mutex);
            frame = shared_frame.clone();
            new_frame_ready = false;
        }

        if (use_yolo)
            annotate_with_depth(frame, current_depth, current_detections);

        if(use_yolo)cv::imshow("YOLO + MiDaS", frame);

        if (!current_depth.empty()) {
            cv::Mat depth_vis;
            cv::normalize(current_depth, depth_vis, 0, 255, cv::NORM_MINMAX);
            depth_vis.convertTo(depth_vis, CV_8U);
            cv::applyColorMap(depth_vis, depth_vis, cv::COLORMAP_MAGMA);
            draw_mean_slope_arrow_sobel(depth_vis, current_depth);
            cv::imshow("Depth", depth_vis);
        }

        if (cv::waitKey(1) == 27) break;

        auto t_end = std::chrono::high_resolution_clock::now();
        double frame_time = std::chrono::duration<double, std::milli>(t_end - frame_start_time).count();
        total_time += frame_time;
        frame_count++;

        {
            std::lock_guard<std::mutex> bench_lock(benchmark_mutex);
            benchmark_count++;

            if (benchmark_count % 30 == 0 && use_bench_marking) {
                double avg_total = total_time / 30.0;
                double avg_midas = midas_time_total / 30.0;
                double avg_yolo  = yolo_time_total / 30.0;

                std::cout << std::fixed << std::setprecision(2);
                std::cout << "\n[Benchmark - últimos 30 frames]\n";
                std::cout << "FPS promedio: " << 1000.0 / avg_total << "\n";
                std::cout << "Tiempo total por frame: " << avg_total << " ms\n";
                std::cout << "→ MiDaS: " << avg_midas << " ms\n";
                if (use_yolo)
                    std::cout << "→ YOLO : " << avg_yolo  << " ms\n";
                std::cout << "\033[2J\033[1;1H";

                total_time = midas_time_total = yolo_time_total = 0.0;
            }
        }
    }

    stop_flag = true;
    frame_cv.notify_all();
    t_capture.join();
    t_midas.join();
    if (use_yolo) t_yolo.join();

    if (use_tcp && sock >= 0) close(sock);
    if (!use_tcp) cap.release();
    cv::destroyAllWindows();
    return 0;
}
