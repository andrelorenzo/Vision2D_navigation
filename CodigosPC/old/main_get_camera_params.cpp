#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <termios.h>
#include <fcntl.h>

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

// Configura la terminal para lectura no bloqueante (tecla sin Enter)
void setNonBlockingTerminal(bool enable) {
    static struct termios oldt, newt;
    if (enable) {
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        fcntl(STDIN_FILENO, F_SETFL, O_NONBLOCK);
    } else {
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    }
}

int main() {
    const int num_images = 10;
    const cv::Size pattern_size(8, 6);
    const float square_size = 0.028f;

    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        std::cerr << "Error al crear socket TCP.\n";
        return 1;
    }

    sockaddr_in server{};
    server.sin_family = AF_INET;
    server.sin_port = htons(SERVER_PORT);
    inet_pton(AF_INET, SERVER_IP, &server.sin_addr);

    std::cout << "Conectando a la cámara por TCP...\n";
    while (connect(sock, (sockaddr*)&server, sizeof(server)) < 0) {
        std::cerr << "Fallo en la conexión, reintentando...\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    std::cout << "Conectado.\n";

    std::vector<std::vector<cv::Point3f>> object_points;
    std::vector<std::vector<cv::Point2f>> image_points;

    std::vector<cv::Point3f> objp;
    for (int i = 0; i < pattern_size.height; ++i)
        for (int j = 0; j < pattern_size.width; ++j)
            objp.emplace_back(j * square_size, i * square_size, 0.0f);

    int collected = 0;
    std::cout << "Vista previa activa. Pulsa Enter para capturar, ESC para salir.\n";

    setNonBlockingTerminal(true); // activar modo no bloqueante de lectura

    while (collected < num_images) {
        cv::Mat frame = get_frame_from_tcp(sock);
        if (frame.empty()) continue;

        cv::imshow("Vista previa", frame);
        int key = cv::waitKey(1); // muestra imagen

        char ch;
        if (read(STDIN_FILENO, &ch, 1) > 0) {
            if (ch == 27) break; // ESC
            if (ch == '\n') {
                cv::Mat gray;
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                std::vector<cv::Point2f> corners;

                bool found = cv::findChessboardCorners(gray, pattern_size, corners,
                                cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);

                if (found) {
                    cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                                     cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
                    image_points.push_back(corners);
                    object_points.push_back(objp);
                    cv::drawChessboardCorners(frame, pattern_size, corners, found);
                    std::cout << "Captura válida " << ++collected << "/" << num_images << "\n";
                } else {
                    std::cout << "Patrón no detectado.\n";
                }
            }
        }
    }

    setNonBlockingTerminal(false); // restaurar terminal
    close(sock);
    cv::destroyAllWindows();

    if (image_points.size() < 3) {
        std::cerr << "No se capturaron suficientes imágenes válidas.\n";
        return 1;
    }

    cv::Mat camera_matrix, dist_coeffs;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(object_points, image_points, pattern_size,
                                     camera_matrix, dist_coeffs, rvecs, tvecs);

    std::cout << "\nCalibración completada. RMS error = " << rms << "\n";
    std::cout << "Matriz de cámara:\n" << camera_matrix << "\n";
    std::cout << "Coeficientes de distorsión:\n" << dist_coeffs << "\n";

    return 0;
}
