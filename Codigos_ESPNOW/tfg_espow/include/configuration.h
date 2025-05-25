
#ifndef CONFIGURATION_H
#define CONFIGURATION_H

// Camera pins configuration for ESP32-CAM
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5

#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22


// WiFi configuration
#define WIFI_SSID "ESP32-CAM"
#define WIFI_PASS "12345678"
#define TCP_PORT 8888

// Serial configuration
#define SERIAL_BR 115200

// Comands enumeration
enum Comands{
    CMD_CAPT = 0x01 // Capture image command
};
#endif // CONFIGURATION_H