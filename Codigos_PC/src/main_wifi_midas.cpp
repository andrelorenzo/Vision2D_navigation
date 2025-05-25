#include <winsock2.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

#pragma comment(lib, "ws2_32.lib")

#define SERVER_IP   "192.168.4.1"
#define SERVER_PORT 8888
#define CAPTURE_CMD "CAPT"

const float CONFIDENCE_THRESHOLD = 0.4;
const float SCORE_THRESHOLD = 0.2;
const float NMS_THRESHOLD = 0.4;
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
    float depth;
};

std::vector<std::string> load_class_list(const std::string& filename) {
    std::vector<std::string> class_list;
    std::ifstream ifs(filename);
    std::string line;
    while (std::getline(ifs, line)) class_list.push_back(line);
    return class_list;
}

cv::Mat format_yolov5(const cv::Mat& source) {
    int col = source.cols;
    int row = source.rows;
    int _max = std::max(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

float get_depth_in_bbox(const cv::Mat& depth_map, const cv::Rect& box) {
    cv::Rect roi = box & cv::Rect(0, 0, depth_map.cols, depth_map.rows);
    cv::Mat region = depth_map(roi);
    cv::Scalar avg = cv::mean(region);
    return static_cast<float>(avg[0]);
}

void detect_with_depth(cv::Mat& image, cv::dnn::Net& yolo_net, cv::dnn::Net& midas_net,
                       std::vector<Detection>& output, const std::vector<std::string>& class_list) {
    // Depth estimation (MiDaS)
    cv::Mat midas_input;
    cv::resize(image, midas_input, cv::Size(256, 256));
    midas_input.convertTo(midas_input, CV_32F, 1.0 / 255.0);
    cv::Mat blob_midas = cv::dnn::blobFromImage(midas_input);
    midas_net.setInput(blob_midas);
    cv::Mat depth = midas_net.forward().reshape(1, 256);
    cv::resize(depth, depth, image.size());

    // Object detection (YOLO)
    cv::Mat blob;
    auto input_image = format_yolov5(image);
    cv::dnn::blobFromImage(input_image, blob, 1./255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(), true, false);
    yolo_net.setInput(blob);
    std::vector<cv::Mat> outputs;
    yolo_net.forward(outputs, yolo_net.getUnconnectedOutLayersNames());

    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float* data = (float*)outputs[0].data;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (int i = 0; i < 25200; ++i) {
        float confidence = data[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float* classes_scores = data + 5;
            cv::Mat scores(1, class_list.size(), CV_32FC1, classes_scores);
            cv::Point class_id_point;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id_point);
            if (max_class_score > SCORE_THRESHOLD) {
                float x = data[0], y = data[1], w = data[2], h = data[3];
                int left = int((x - 0.5 * w) * x_factor);
                int top = int((y - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(confidence);
                class_ids.push_back(class_id_point.x);
            }
        }
        data += 85;
    }

    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
    for (int i : nms_result) {
        float depth_val = get_depth_in_bbox(depth, boxes[i]);
        output.push_back({class_ids[i], confidences[i], boxes[i], depth_val});
    }
}

bool recvAll(SOCKET sock, uint8_t* buffer, size_t length) {
    size_t total = 0;
    while (total < length) {
        int bytes = recv(sock, reinterpret_cast<char*>(buffer + total), length - total, 0);
        if (bytes <= 0) return false;
        total += bytes;
    }
    return true;
}

int main() {
    WSADATA wsa;
    WSAStartup(MAKEWORD(2, 2), &wsa);
    SOCKET sock = socket(AF_INET, SOCK_STREAM, 0);

    sockaddr_in server{};
    server.sin_family = AF_INET;
    server.sin_port = htons(SERVER_PORT);
    server.sin_addr.s_addr = inet_addr(SERVER_IP);

    std::cout << "Conectando a la camera..." << std::endl;
    while (connect(sock, reinterpret_cast<sockaddr*>(&server), sizeof(server)) < 0) {
        std::cerr << "No se pudo conectar reintentando...\n";
        std::cout.flush();
        std::cout << "\033[2J\033[H"; // Clear console
        Sleep(500);
    }

    std::vector<std::string> class_list = load_class_list("C:/Users/andre/Documents/Trabajo_fin_master/Codigos_PC/src/classes.txt");
    cv::dnn::Net yolo_net = cv::dnn::readNet("C:/Users/andre/Documents/Trabajo_fin_master/Codigos_PC/src/yolov5n.onnx");
    yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    cv::dnn::Net midas_net = cv::dnn::readNet("C:/Users/andre/Documents/Trabajo_fin_master/Codigos_PC/src/midas_v21_small_256.onnx");
    midas_net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    midas_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

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

        std::vector<Detection> detections;
        detect_with_depth(img, yolo_net, midas_net, detections, class_list);

        for (const auto& d : detections) {
            cv::rectangle(img, d.box, cv::Scalar(0, 255, 0), 2);
            std::ostringstream label;
            label << class_list[d.class_id] << " " << std::fixed << std::setprecision(2) << d.confidence << " D:" << d.depth;
            cv::putText(img, label.str(), cv::Point(d.box.x, d.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 2);
        }

        cv::imshow("YOLO + MiDaS - ESP32-CAM", img);
        if (cv::waitKey(1) == 27) break;
    }

    closesocket(sock);
    WSACleanup();
    return 0;
}
