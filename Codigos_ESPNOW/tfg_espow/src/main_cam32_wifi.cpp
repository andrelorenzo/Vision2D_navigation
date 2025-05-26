#define ESP32_CAM_WIFI

#ifdef ESP32_CAM_WIFI
#include <Arduino.h>
#include <WiFi.h>
#include <esp_camera.h>
#include "configuration.h"



WiFiServer server(TCP_PORT);

void setupCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;

  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;

  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size = FRAMESIZE_VGA;
    config.jpeg_quality = 12;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
  }else
  {
    Serial.println("Camera initialized successfully");
  }
}

bool readExact(WiFiClient& client, uint8_t* buffer, size_t length, uint32_t timeout_ms = 1000) {
  uint32_t start = millis();
  size_t received = 0;
  while (received < length && (millis() - start < timeout_ms)) {
    if (client.available()) {
      int c = client.read();
      if (c >= 0) buffer[received++] = (uint8_t)c;
    }
  }
  return received == length;
}

void setup() {
  delay(2000); // Estabilizar alimentación
  Serial.begin(SERIAL_BR);
  setupCamera();
  delay(500); // Dar tiempo al sistema antes del WiFi

  WiFi.softAP(WIFI_SSID, WIFI_PASS);
  server.begin();
}

void loop() {
  WiFiClient client = server.available();
  if (client) {
    //Serial.println("Cliente conectado");

    while (client.connected()) {
      if (client.available() >= 4) {
        uint8_t cmd[4];
        if (!readExact(client, cmd, 4)) {
          Serial.println("Fallo leyendo comando completo.");
          break;
        }

        if (memcmp(cmd, "CAPT", 4) == 0) {
          Serial.println("Comando CAPT recibido. Capturando imagen...");

          camera_fb_t* fb = esp_camera_fb_get();
          if (!fb || fb->len == 0) {
            //Serial.println("Error al capturar imagen");
            continue;
          }

          uint32_t size = fb->len;
          uint8_t sizeBytes[4] = {
            (uint8_t)((size >> 24) & 0xFF),
            (uint8_t)((size >> 16) & 0xFF),
            (uint8_t)((size >> 8) & 0xFF),
            (uint8_t)(size & 0xFF)
          };

          client.write(sizeBytes, 4);
          size_t sent = client.write(fb->buf, fb->len);

          Serial.printf("Imagen enviada (%u/%u bytes)\n", sent, fb->len);
          esp_camera_fb_return(fb);
        }
      }

      delay(10);  // pequeña pausa para evitar sobrecarga
    }

    client.stop();
    //Serial.println("Cliente desconectado");
  }
}
#endif
