#include <WiFi.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include "Adafruit_MQTT.h"
#include "Adafruit_MQTT_Client.h"

// === WiFi & MQTT Info ===
const char* ssid = "A1 016";
const char* password = "20454418";

// MQTT broker info
#define MQTT_SERVER      "industrial.api.ubidots.com"
#define MQTT_SERVERPORT  1883
#define MQTT_USERNAME    "BBUS-6L3O4cSRsFqpUEW4e5dJNVxCyiOwJa"  // optional
#define MQTT_KEY         ""  // optional

WiFiClient client;
Adafruit_MQTT_Client mqtt(&client, MQTT_SERVER, MQTT_SERVERPORT, MQTT_USERNAME, MQTT_KEY);

// MQTT topic
Adafruit_MQTT_Publish fall_data = Adafruit_MQTT_Publish(&mqtt, "/v1.6/devices/esp32-uni159");

Adafruit_MPU6050 mpu;

void connectToWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println(" connected!");
}

void connectToMQTT() {
  int8_t ret;
  while ((ret = mqtt.connect()) != 0) {
    Serial.print("MQTT connection failed, code: ");
    Serial.println(mqtt.connectErrorString(ret));
    Serial.println("Retrying MQTT in 5 seconds...");
    mqtt.disconnect();
    delay(5000);
  }
  Serial.println("MQTT Connected!");
}

void setup() {
  Serial.begin(115200);
  Wire.begin();

  connectToWiFi();

  if (!mpu.begin()) {
    Serial.println("MPU6050 not found!");
    while (1) delay(10);
  }

  Serial.println("MPU6050 Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
}

void loop() {
  if (!mqtt.connected()) {
    connectToMQTT();
  }
  mqtt.processPackets(10);
  mqtt.ping();

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  char payload[200];
  snprintf(payload, sizeof(payload),
           "{\"Ax\":%.2f,\"Ay\":%.2f,\"Az\":%.2f,\"Gx\":%.2f,\"Gy\":%.2f,\"Gz\":%.2f}",
           a.acceleration.x, a.acceleration.y, a.acceleration.z,
           g.gyro.x, g.gyro.y, g.gyro.z);

  Serial.println(payload);
  if (!fall_data.publish(payload)) {
    Serial.println("Failed to publish");
  }

  delay(1000);
}