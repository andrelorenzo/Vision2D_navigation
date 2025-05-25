#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "serialib.h"

#define HEADER1 0xA5
#define HEADER2 0x5A

bool waitForHeader(serialib& serial, unsigned long timeout_ms = 3000) {
    char b;
    unsigned long start = GetTickCount();

    while (GetTickCount() - start < timeout_ms) {
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
    unsigned long start = GetTickCount();
    size_t totalRead = 0;

    while (totalRead < size && GetTickCount() - start < timeout_ms) {
        int read = serial.readBytes((char*)&buffer[totalRead], size - totalRead, 100);
        if (read < 0) return false;
        totalRead += read;
    }

    return totalRead == size;
}

int main() {
    serialib serial;
    const char* port = "COM3";  // Cambia si es necesario

    if (serial.openDevice(port, 250000) != 1) {
        std::cerr << "No se pudo abrir el puerto: " << port << std::endl;
    }

    while (true) {
        std::cout << "\nEsperando cabecera..." << std::endl;
        if (!waitForHeader(serial)) {
            std::cerr << "No se detecto cabecera valida. Reintentando...\n";
            continue;
        }

        char sizeBytes[2];
        if (serial.readBytes(sizeBytes, 2, 500) != 2) {
            std::cerr << "Error leyendo tamaño de imagen. Reintentando...\n";
            continue;
        }

        int size = ((uint8_t)sizeBytes[0] << 8) | (uint8_t)sizeBytes[1];
        std::cout << "Tamano de imagen recibido: " << size << " bytes" << std::endl;

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

        cv::imshow("Imagen recibida", img);
        int key = cv::waitKey(1);
        if (key == 27) break;
    }

    serial.closeDevice();
    return 0;
}
