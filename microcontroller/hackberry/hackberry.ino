#include <Adafruit_VC0706.h>
#include <SoftwareSerial.h>         
#include <AltSoftSerial.h>

#define CAMERA_TX 2
#define CAMERA_RX 3
#define BLUETOOTH_TX 8
#define BLUETOOTH_RX 9
#define BLUETOOTH_STATE 4
#define BLUETOOTH_EN 3
#define BAUD_RATE 9600
#define BUFF_SIZE 32
// Using SoftwareSerial (Arduino 1.0+) or NewSoftSerial (Arduino 0023 & prior):
#if ARDUINO >= 100
// On Uno: camera TX connected to pin 2, camera RX to pin 3:
SoftwareSerial camera_connection = SoftwareSerial(CAMERA_TX, CAMERA_RX);
#else
NewSoftSerial camera_connection = NewSoftSerial(CAMERA_TX, CAMERA_RX);
#endif

AltSoftSerial btSerial = AltSoftSerial(BLUETOOTH_TX, BLUETOOTH_RX);

Adafruit_VC0706 cam = Adafruit_VC0706(&camera_connection);


int setup_camera(void) {

  // Try to locate the camera
  if (cam.begin()) {
    Serial.println("Camera Found");
  } else {
    Serial.println("Failed, Camera Not Found");
    return -1;
  } 

  delay(100);

  /* Choose camera setting */

  /*cam.setImageSize(VC0706_640x480);        // biggest*/
  cam.setImageSize(VC0706_320x240);        // medium
  /*cam.setImageSize(VC0706_160x120);        // small*/

  return 0;

}

int setup_bluetooth() {

  // Bluetooth initialization:
  btSerial.begin(BAUD_RATE);

  // Define pins
  pinMode(BLUETOOTH_STATE, INPUT);
  pinMode(BLUETOOTH_EN, OUTPUT);

  delay(100);

  if (btSerial.isListening()) {
    Serial.println("Bluetooth Found");
    return 0;
  } else {
    Serial.println("Failed, Bluetooth Not Found");
    return -1;
  }

}

void take_picture() {

  int32_t large_delay_after_write = 50;
  int32_t small_delay_after_write = 10;

  if (!cam.takePicture()) {
    Serial.println("Failed to snap!");
    return;
  }

  // Get the size of the image (frame) taken  
  uint16_t jpglen = cam.frameLength();
  uint16_t frame_size = jpglen;
  uint8_t jpglen_array[2];

  // Copy higher 8 bits
  jpglen_array[0] = jpglen >> 8;

  // Copy lower 8 bits
  jpglen_array[1] = jpglen;

  Serial.print("Image size = ");
  Serial.print(jpglen, HEX);
  Serial.print(" = ");
  Serial.println(jpglen, DEC);

  int32_t time = millis();

  // Wait for send command
  while (!btSerial.available() && btSerial.read() != '1') {}

  btSerial.write(jpglen_array, sizeof(jpglen_array));

  delay(small_delay_after_write);

  while (jpglen > 0) {

    uint8_t * buffer;

    // Read 32 bytes at a time;
    uint8_t bytesToRead = min(BUFF_SIZE, jpglen); 
    buffer = cam.readPicture(bytesToRead);

    delay(small_delay_after_write);

    // Write byte stream to serial port
    btSerial.write(buffer, bytesToRead);

    delay(large_delay_after_write);

    // Subtract sent bytes
    jpglen -= bytesToRead;

  }

  time = millis() - time;
  int32_t total_delay = (frame_size/BUFF_SIZE)*(large_delay_after_write \
      + small_delay_after_write) \
      + small_delay_after_write;

  Serial.print("Total time = ");
  Serial.print(time); 
  Serial.print(" ms, Delay time = ");
  Serial.print(total_delay); 
  Serial.print(" ms, Transfer time = ");
  Serial.println(time-total_delay); 

}

void setup() {

  Serial.begin(BAUD_RATE);
  Serial.println("-------[ Hackberry ]-------");
  
  if (setup_camera() != 0) {
    return;
  } 

  if (setup_bluetooth() != 0) {
    return;
  } 

  while(!Serial) {}

  delay(300);

}

void loop() {
  cam.reset();
  delay(300);
  take_picture();
}

