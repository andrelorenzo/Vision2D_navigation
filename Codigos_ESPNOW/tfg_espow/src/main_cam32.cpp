#ifdef ESP32_CAM
#include <WiFi.h>
#include <esp_now.h>
#include <esp_camera.h>
#include <SPIFFS.h>
#include "Arduino.h"
#include "camera_pins.h"

// ESP-NOW
uint8_t master_mac[] = {0xA0, 0xB7, 0x65, 0x4A, 0x7B, 0xD8};
#define CHANNEL 1
#define PACKET_SIZE 200
#define HEADER_SIZE 6
#define MAX_IMAGE_SIZE 60000
enum SendState { IDLE, SENDING };

File imageFile;
uint16_t image_bytes = 0;
uint8_t total_msg = 0;
uint8_t current_msg = 0;
SendState send_state = IDLE;
volatile bool send_allowed = true;
unsigned long last_image_sent_time = 0;


bool image_ready = false;

void capture_image_and_send() {
  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb || fb->len == 0) {
    Serial.println("[ERROR] Captura de imagen fallida o vacía");
    return;
  }

  if (fb->len > MAX_IMAGE_SIZE) {
    Serial.printf("[ERROR] Imagen demasiado grande: %u bytes\n", fb->len);
    esp_camera_fb_return(fb);
    return;
  }

  File file = SPIFFS.open("/frame.jpg", FILE_WRITE);
  if (!file) {
    Serial.println("[ERROR] No se pudo abrir el archivo para guardar la imagen");
    esp_camera_fb_return(fb);
    return;
  }

  file.write(fb->buf, fb->len);
  file.close();

  image_bytes = fb->len;
  total_msg = (image_bytes + PACKET_SIZE - 1) / PACKET_SIZE;
  current_msg = 0;

  esp_camera_fb_return(fb);

  imageFile = SPIFFS.open("/frame.jpg", FILE_READ);
  if (!imageFile) {
    Serial.println("[ERROR] No se pudo abrir archivo para envío");
    return;
  }

  send_state = SENDING;
  send_allowed = true;
}

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  send_allowed = true;
  // Serial.print("ACK de envío => ");
  // Serial.println(status == ESP_NOW_SEND_SUCCESS ? "ÉXITO" : "FALLO");
}

void onDataRecv(const uint8_t* mac, const uint8_t* data, int len) {
  if (len == 4 && memcmp(data, "CAPT", 4) == 0) {
    unsigned long now = millis();
    if (last_image_sent_time > 0) {
      unsigned long elapsed = now - last_image_sent_time;
      Serial.printf("[DATA RATE] Tiempo entre imágenes: %lu ms\n", elapsed);
    }

    // Serial.println("Comando CAPT recibido");
    capture_image_and_send();
  }
}

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
  config.pin_sccb_sda  = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  if (psramFound()) {
    config.frame_size = FRAMESIZE_QVGA;
    config.jpeg_quality = 10;
    config.fb_count = 2;
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    config.frame_size = FRAMESIZE_QQVGA;
    config.jpeg_quality = 15;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
  } else {
    Serial.println("Camera initialized successfully");
  }
}

void setup() {
  Serial.begin(115200);
  if (!SPIFFS.begin(true)) {
    Serial.println("Error iniciando SPIFFS");
    return;
  }

  setupCamera();

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error inicializando ESP-NOW");
    return;
  }

  esp_now_register_recv_cb(onDataRecv);
  esp_now_register_send_cb(OnDataSent);

  esp_now_peer_info_t peerInfo = {};
  memset(&peerInfo, 0, sizeof(esp_now_peer_info_t));
  memcpy(peerInfo.peer_addr, master_mac, 6);
  peerInfo.channel = CHANNEL;
  peerInfo.encrypt = false;

  esp_err_t result = esp_now_add_peer(&peerInfo);
  if (result != ESP_OK) {
    Serial.printf("Error añadiendo peer: %d\n", result);
  }
}
unsigned long last_send_time = 0;  // nueva variable global
unsigned long time_between_send = 15;
void loop() {
  if (send_state == SENDING) {
    if (!imageFile) {
      Serial.println("[ERROR] imageFile no está abierto");
      send_state = IDLE;
      return;
    }

    if (current_msg >= total_msg) {
      // Serial.println("=== Todos los paquetes enviados ===");
      imageFile.close();
      last_image_sent_time = millis();
      send_state = IDLE;
      return;
    }

    // Intentar enviar si permitido
    if (send_allowed) {
      uint8_t packet[PACKET_SIZE + HEADER_SIZE];
      packet[0] = current_msg;
      packet[1] = total_msg;
      packet[2] = (image_bytes >> 8) & 0xFF;
      packet[3] = image_bytes & 0xFF;

      imageFile.seek(current_msg * PACKET_SIZE);
      size_t readBytes = imageFile.read(packet + HEADER_SIZE, PACKET_SIZE);

      if (readBytes == 0) {
        Serial.printf("[ERROR] No se pudieron leer datos para el paquete %u\n", current_msg);
        send_state = IDLE;
        imageFile.close();
        return;
      }

      esp_err_t result = esp_now_send(master_mac, packet, readBytes + HEADER_SIZE);
      // Serial.printf("Enviando paquete %u/%u: ", current_msg + 1, total_msg);

      if (result == ESP_OK) {
        // Serial.println("OK");
        send_allowed = false;
        last_send_time = millis();  // iniciar timeout
        current_msg++;
      } else {
        Serial.printf("FALLO (0x%X)\n", result);
        send_allowed = true;
      }
    }
    // Timeout para forzar reactivación
    else if (millis() - last_send_time > time_between_send) {
      send_allowed = true;
    }
  }
    // --- Comandos por Serial ---
  if (Serial.available()) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command.equalsIgnoreCase("")) {
      Serial.println("Comando CAPT recibido por Serial");
      capture_image_and_send();
    } else if (command.startsWith("t:")) {
      int newTimeout = command.substring(2).toInt();
      if (newTimeout > 0) {
        time_between_send = newTimeout;
        Serial.printf("Nuevo timeout entre paquetes: %lu ms\n", time_between_send);
      } else {
        Serial.println("[ERROR] Valor de timeout inválido");
      }
    } else if (command.equalsIgnoreCase("s")) {
      Serial.println("===== ESTADO ACTUAL =====");
      Serial.printf("image_bytes: %u\n", image_bytes);
      Serial.printf("total_msg:   %u\n", total_msg);
      Serial.printf("current_msg: %u\n", current_msg);
      Serial.print("send_state:  ");
      Serial.println(send_state == IDLE ? "IDLE" : "SENDING");
      Serial.printf("send_allowed: %s\n", send_allowed ? "true" : "false");
      Serial.printf("time_between_send: %lu ms\n", time_between_send);
      Serial.println("==========================");
    }
  }

}



#endif
