#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>
#include <chrono>

#include "yolov11.hpp"

#define SERVER_IP   "192.168.4.1"
#define SERVER_PORT 8888
#define CAPTURE_CMD "CAPT"

bool recvAll(int sock, uint8_t* buffer, size_t length) {
    size_t total = 0;
    while (total < length) {
        ssize_t bytes = recv(sock, buffer + total, length - total, 0);
        if (bytes <= 0) return false;
        total += bytes;
    }
    return true;
}

int main(int argc, char** argv) {
    bool use_yolo = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--yolo") {
            use_yolo = true;
        }
    }

    std::cout << "[INFO] YOLO activado: " << (use_yolo ? "Sí" : "No") << std::endl;

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Error al crear el socket.\n";
        return 1;
    }

    sockaddr_in server{};
    server.sin_family = AF_INET;
    server.sin_port = htons(SERVER_PORT);
    inet_pton(AF_INET, SERVER_IP, &server.sin_addr);

    std::cout << "Conectando a la cámara..." << std::endl;
    if (connect(sock, (sockaddr*)&server, sizeof(server)) < 0) {
        std::cerr << "No se pudo conectar.\n";
        return 1;
    }

    YOLOv11 model("../models/yolo/yolo11n.onnx");

    std::chrono::steady_clock::time_point last_frame_time = std::chrono::steady_clock::now();
    int frame_count = 0;
    double total_elapsed_ms = 0.0;

    while (true) {
        send(sock, CAPTURE_CMD, 4, 0);

        uint8_t sizeBytes[4];
        if (!recvAll(sock, sizeBytes, 4)) break;
        int imgSize = (sizeBytes[0] << 24) | (sizeBytes[1] << 16) | (sizeBytes[2] << 8) | sizeBytes[3];
        if (imgSize <= 0 || imgSize > 150000) break;

        std::vector<uint8_t> imgBuffer(imgSize);
        if (!recvAll(sock, imgBuffer.data(), imgSize)) break;

        auto start = std::chrono::steady_clock::now();

        cv::Mat img = cv::imdecode(imgBuffer, cv::IMREAD_COLOR);
        if (img.empty()) continue;

        if (use_yolo) {
            std::vector<ObjectBBox> bbox_list = model.detect(img);
            for (auto& bbox : bbox_list) {
                bbox.draw(img);
            }
        }

        auto end = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - last_frame_time).count();
        last_frame_time = end;

        total_elapsed_ms += elapsed_ms;
        frame_count++;

        double instant_fps = (elapsed_ms > 0) ? 1000.0 / elapsed_ms : 0.0;
        // std::cout << std::fixed << std::setprecision(2)
           //       << "Tiempo por frame: " << elapsed_ms << " ms | FPS: " << instant_fps << std::endl;

        if (frame_count % 20 == 0) {
            double avg_ms = total_elapsed_ms / 20.0;
            double avg_fps = (avg_ms > 0) ? 1000.0 / avg_ms : 0.0;
            std::cout << "----- Promedio (20 frames): " << avg_ms << " ms/frame | " << avg_fps << " FPS -----" << std::endl;
            total_elapsed_ms = 0.0;  // Reset acumulador
        }

        std::cout << "\033[2J\033[1;1H";  // Limpiar pantalla (opcional)

        cv::imshow("Detección YOLO - ESP32-CAM", img);
        if (cv::waitKey(1) == 27) break;
    }

    close(sock);
    return 0;
}
