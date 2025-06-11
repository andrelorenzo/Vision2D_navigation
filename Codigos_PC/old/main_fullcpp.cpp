#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <cstring>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "yolov11.hpp"

#define SERVER_IP   "192.168.4.1"
#define SERVER_PORT 8888
#define CAPTURE_CMD "CAPT"

int frame_count = 0;
double total_time = 0.0, yolo_time = 0.0, midas_time = 0.0;


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


        cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
        if (roi.area() <= 0) continue;

        cv::Mat depth_roi = depth_map(roi);
        float distance = static_cast<float>(cv::mean(depth_roi)[0]);

        std::ostringstream label;
        label << bbox.label << " " << std::fixed << std::setprecision(2)
              << bbox.conf << " D:" << distance;

        bbox.draw(frame);
        cv::putText(frame, label.str(), cv::Point(x1, y1 - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 2);
    }
}

int main(int argc, char** argv) {
    bool use_tcp = true;
    if (argc > 1 && std::string(argv[1]) == "--camera") {
        use_tcp = false;
    }

    // === Cargar modelo YOLOv11 ===
    YOLOv11 model("../models/yolo/yolo11n.onnx", 0.45f, 0.45f,
        [](int lbl_id, const std::string& lbl) {
            return lbl_id >= 0;
        });

    // === Cargar modelo MiDaS ===
    cv::dnn::Net midas = cv::dnn::readNet("../models/midas/model-small.onnx");
    midas.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    midas.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    int sock = -1;
    cv::VideoCapture cap;

    if (use_tcp) {
        sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            std::cerr << "Error al crear el socket\n";
            return 1;
        }

        sockaddr_in server{};
        server.sin_family = AF_INET;
        server.sin_port = htons(SERVER_PORT);
        inet_pton(AF_INET, SERVER_IP, &server.sin_addr);

        std::cout << "Conectando a la cámara por TCP..." << std::endl;
        while (connect(sock, (sockaddr*)&server, sizeof(server)) < 0) {
            std::cerr << "Fallo en la conexión, reintentando...\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        std::cout << "Conectado." << std::endl;
    } else {
        std::cout << "Usando la cámara local (/dev/video0)" << std::endl;
        cap.open(0);
        if (!cap.isOpened()) {
            std::cerr << "No se pudo abrir la cámara local\n";
            return 1;
        }
    }

    while (true) {
        auto t_start = std::chrono::high_resolution_clock::now();

        cv::Mat frame = use_tcp ? get_frame_from_tcp(sock) : get_frame_from_camera(cap);
        if (frame.empty()) continue;

        // --- MiDaS ---
        auto t_midas_start = std::chrono::high_resolution_clock::now();
        cv::Mat depth_map = estimate_depth_map(midas, frame);
        auto t_midas_end = std::chrono::high_resolution_clock::now();

        // --- YOLO ---
        auto t_yolo_start = std::chrono::high_resolution_clock::now();
        std::vector<ObjectBBox> bbox_list = model.detect(frame);
        auto t_yolo_end = std::chrono::high_resolution_clock::now();

        annotate_with_depth(frame, depth_map, bbox_list);

        // Visualización
        cv::imshow("YOLO + Depth", frame);

        cv::Mat depth_vis;
        cv::normalize(depth_map, depth_vis, 0, 255, cv::NORM_MINMAX);
        depth_vis.convertTo(depth_vis, CV_8U);
        cv::applyColorMap(depth_vis, depth_vis, cv::COLORMAP_MAGMA);
        cv::imshow("Depth", depth_vis);

        if (cv::waitKey(1) == 27) break;

        auto t_end = std::chrono::high_resolution_clock::now();

        // --- Acumular tiempos ---
        double total = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        double midas_dur = std::chrono::duration<double, std::milli>(t_midas_end - t_midas_start).count();
        double yolo_dur  = std::chrono::duration<double, std::milli>(t_yolo_end  - t_yolo_start).count();

        total_time += total;
        midas_time += midas_dur;
        yolo_time  += yolo_dur;
        frame_count++;

        // --- Imprimir promedio cada 30 frames ---
        if (frame_count % 30 == 0) {
            double avg_total = total_time / 30.0;
            double avg_midas = midas_time / 30.0;
            double avg_yolo  = yolo_time  / 30.0;

            std::cout << std::fixed << std::setprecision(2);
            std::cout << "\n[Benchmark - últimos 30 frames]\n";
            std::cout << "FPS promedio: " << 1000.0 / avg_total << "\n";
            std::cout << "Tiempo total por frame: " << avg_total << " ms\n";
            std::cout << "→ MiDaS: " << avg_midas << " ms\n";
            std::cout << "→ YOLO : " << avg_yolo  << " ms\n";
            std::cout << "\033[2J\033[1;1H";    

            total_time = midas_time = yolo_time = 0.0;
        }
    }


    if (use_tcp && sock >= 0) close(sock);
    if (!use_tcp) cap.release();
    cv::destroyAllWindows();
    return 0;
}
