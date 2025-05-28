#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

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

int main() {
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

    YOLOv11 model("../models/yolo11n.onnx");

    while (true) {
        send(sock, CAPTURE_CMD, 4, 0);
        uint8_t sizeBytes[4];
        if (!recvAll(sock, sizeBytes, 4)) break;
        int imgSize = (sizeBytes[0] << 24) | (sizeBytes[1] << 16) | (sizeBytes[2] << 8) | sizeBytes[3];
        if (imgSize <= 0 || imgSize > 150000) break;

        std::vector<uint8_t> imgBuffer(imgSize);
        if (!recvAll(sock, imgBuffer.data(), imgSize)) break;

        cv::Mat img = cv::imdecode(imgBuffer, cv::IMREAD_COLOR);
        if (img.empty()) continue;

        std::vector<ObjectBBox> bbox_list = model.detect(img);
        for (auto& bbox : bbox_list) {
            std::cout << "Label:" << bbox.label << " Conf: " << bbox.conf;
            std::cout << "(" << bbox.x1 << ", " << bbox.y1 << ") ";
            std::cout << "(" << bbox.x2 << ", " << bbox.y2 << ")" << std::endl;
            bbox.draw(img);
        }

        cv::imshow("Detección YOLO - ESP32-CAM", img);
        if (cv::waitKey(1) == 27) break;
    }

    close(sock);
    return 0;
}
