#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/opencv.hpp>
#include "serialib.h"

#define HEADER1 0xA5
#define HEADER2 0x5A
#define SERIAL_PORT "/dev/ttyUSB0"
#define BAUDRATE 115200

// Devuelve tiempo actual en milisegundos
unsigned long getTimeMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool waitForHeader(serialib& serial, unsigned long timeout_ms = 3000) {
    char b;
    unsigned long start = getTimeMs();

    while (getTimeMs() - start < timeout_ms) {
        if (serial.readChar(&b, 50) != 1) continue;
        if ((uint8_t)b == HEADER1) {
            if (serial.readChar(&b, 50) != 1) continue;
            if ((uint8_t)b == HEADER2) return true;
        }
    }
    return false;
}

bool readBytes(serialib& serial, std::vector<uint8_t>& buffer, size_t size, unsigned long timeout_ms = 5000) {
    buffer.resize(size);
    unsigned long start = getTimeMs();
    size_t totalRead = 0;

    while (totalRead < size && getTimeMs() - start < timeout_ms) {
        int read = serial.readBytes((char*)&buffer[totalRead], size - totalRead, 100);
        if (read < 0) return false;
        totalRead += read;
    }

    return totalRead == size;
}

int main() {
    serialib serial;
    const char* port = SERIAL_PORT;

    int result = serial.openDevice(SERIAL_PORT, BAUDRATE);
    if (result != 1) {
        std::cerr << "Error al abrir el puerto: " << SERIAL_PORT << ", código: " << result << std::endl;
        return 1;
    }

    std::chrono::steady_clock::time_point lastFrameTime = std::chrono::steady_clock::now();

    while (true) {
        // std::cout << "\nEsperando cabecera..." << std::endl;
        if (!waitForHeader(serial)) {
            std::cerr << "No se detectó cabecera válida. Reintentando...\n";
            continue;
        }

        char sizeBytes[2];
        if (serial.readBytes(sizeBytes, 2, 500) != 2) {
            std::cerr << "Error leyendo tamaño de imagen. Reintentando...\n";
            continue;
        }

        int size = ((uint8_t)sizeBytes[0] << 8) | (uint8_t)sizeBytes[1];
        std::cout << "Tamaño de imagen recibido: " << size << " bytes" << std::endl;

        std::vector<uint8_t> imageData;
        if (!readBytes(serial, imageData, size)) {
            std::cerr << "Error recibiendo imagen completa. Reintentando...\n";
            continue;
        }

        cv::Mat img = cv::imdecode(imageData, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "[ERROR] Imagen corrupta o incompleta.\n";
            continue;
        }

        // Medición de tiempo transcurrido
        auto currentFrameTime = std::chrono::steady_clock::now();
        auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(currentFrameTime - lastFrameTime).count();
        lastFrameTime = currentFrameTime;

        double fps = (durationMs > 0) ? (1000.0 / durationMs) : 0.0;
        std::cout << "Tiempo desde último frame: " << durationMs << " ms, FPS: " << fps << std::endl;
        std::cout << "\033[2J\033[1;1H";

        cv::imshow("Imagen recibida", img);
        if (cv::waitKey(1) == 27) break;  // Tecla ESC para salir
    }

    serial.closeDevice();
    return 0;
}
