#ifdef ESP32_BASE

#include "Arduino.h"
#include <WiFi.h>
#include <esp_now.h>
#include <SPIFFS.h>

#define PACKET_SIZE 200
#define HEADER_SIZE 6
#define MAX_IMAGE_SIZE 60000

uint8_t cam_mac[] = {0x5C, 0x01, 0x3B, 0x32, 0x3C, 0x54}; // MAC de la ESP32-CAM
uint8_t image_buffer[MAX_IMAGE_SIZE];
bool fragment_received[300] = {false};  // Hasta 300 fragmentos

uint16_t image_size = 0;
uint8_t expected_fragments = 0;
uint8_t received_fragments = 0;
bool initialized = false;
bool image_complete = false;

void send_capture_command() {
  const char* cmd = "CAPT";
  esp_now_send(cam_mac, (uint8_t*)cmd, 4);
  Serial.println("Comando CAPT enviado a cámara");
}

void OnDataRecv(const uint8_t *mac, const uint8_t *incomingData, int len) {
  if (len < HEADER_SIZE) return;

  uint8_t fragment_id = incomingData[0];
  uint8_t total_fragments = incomingData[1];
  uint16_t total_size = (incomingData[2] << 8) | incomingData[3];

  if (!initialized || total_size != image_size || total_fragments != expected_fragments) {
    memset(image_buffer, 0, MAX_IMAGE_SIZE);
    memset(fragment_received, 0, sizeof(fragment_received));
    image_size = total_size;
    expected_fragments = total_fragments;
    received_fragments = 0;
    initialized = true;
    image_complete = false;
  }

  int offset = fragment_id * PACKET_SIZE;
  int data_len = len - HEADER_SIZE;
  if (offset + data_len > MAX_IMAGE_SIZE) return;

  memcpy(&image_buffer[offset], &incomingData[HEADER_SIZE], data_len);

  if (!fragment_received[fragment_id]) {
    fragment_received[fragment_id] = true;
    received_fragments++;
    // Serial.printf("Fragmento %d/%d recibido (%d bytes)\n", fragment_id + 1, expected_fragments, data_len);
  }

  if (received_fragments == expected_fragments) {
    image_complete = true;
    Serial.println("Imagen completa recibida");
  }
}

void save_image_to_spiffs() {
  File file = SPIFFS.open("/received_frame.jpg", FILE_WRITE);
  if (!file) {
    Serial.println("Error al guardar imagen");
    return;
  }
  file.write(image_buffer, image_size);
  file.close();

  File imageFile = SPIFFS.open("/received_frame.jpg", FILE_READ);
  if (!imageFile) {
    Serial.println("Error al abrir imagen para enviar por Serial");
    return;
  }

  Serial.println("Enviando imagen por puerto serie...");
  Serial.write(0xA5);
  Serial.write(0x5A);
  Serial.write((image_size >> 8) & 0xFF);
  Serial.write(image_size & 0xFF);

  while (imageFile.available()) {
    Serial.write(imageFile.read());
  }

  imageFile.close();
}

void setup() {
  Serial.begin(115200);
  if (!SPIFFS.begin(true)) {
    Serial.println("Error inicializando SPIFFS");
    return;
  }

  WiFi.mode(WIFI_STA);
  WiFi.disconnect();

  if (esp_now_init() != ESP_OK) {
    Serial.println("Error inicializando ESP-NOW");
    return;
  }

  esp_now_register_recv_cb(OnDataRecv);

  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, cam_mac, 6);
  peerInfo.channel = 1;
  peerInfo.encrypt = false;
  esp_now_add_peer(&peerInfo);

  delay(5000);
  send_capture_command();
}

void loop() {
  if (image_complete) {
    save_image_to_spiffs();

    // Reiniciar variables para siguiente captura
    image_complete = false;
    initialized = false;
    image_size = 0;
    expected_fragments = 0;
    received_fragments = 0;

    delay(10);  // breve pausa antes de solicitar nueva
    send_capture_command();
  }
}

#endif // ESP32_BASE
