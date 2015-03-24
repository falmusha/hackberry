  #include <Adafruit_VC0706.h>
#include <SoftwareSerial.h>         
#include <AltSoftSerial.h>

#define CAMERA_TX 2
#define CAMERA_RX 3
#define BLUETOOTH_TX 8
#define BLUETOOTH_RX 9
#define BUFF_SIZE 64
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

  /* Choose camera setting */
  /*cam.setImageSize(VC0706_640x480);        // biggest*/
  cam.setImageSize(VC0706_320x240);        // medium
  /*cam.setImageSize(VC0706_160x120);        // small*/
//  cam.setBaud9600();
//  cam.setBaud19200();
//  cam.setBaud38400();
//  cam.setBaud57600();
  cam.setBaud115200();

  return 0;

}

int setup_bluetooth() {

  // Bluetooth initialization:
  btSerial.begin(57600);
  
  if (!btSerial.isListening()) {
    Serial.println("Failed, Bluetooth Not Found");
    return -1;
  }

  btSerial.print("AT+BAUD7");
  delay(1000);
  
  while(btSerial.available() <= 0) {}

  String response = "";
  while(btSerial.available()) { // While there is more to be read, keep reading.
    response += (char)btSerial.read();
  }
  
  if (response == "OK57600") {
    Serial.println("Bluetooth Found");
  } else {
    Serial.println("Failed, Bluetooth Not Found");
  }
  
}

void wait_on_ack() {
  while (btSerial.available() <= 0) {}
  char b = btSerial.read();
  if (b == '7') {
    Serial.println("Ack received");
  } else {
    Serial.println("NOOO Ack");
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

  btSerial.write(jpglen_array, sizeof(jpglen_array));
  btSerial.flush();
//  wait_on_ack();

  while (jpglen > 0) {

    uint8_t * bf;
    // Read 32 bytes at a time;
    uint8_t bytesToRead = min(BUFF_SIZE, jpglen); 
    bf = cam.readPicture(bytesToRead);

    // Write byte stream to serial port
    btSerial.write(bf, bytesToRead);
    btSerial.flush();
//    wait_on_ack();

    Serial.print(".");
    
    // Subtract sent bytes
    jpglen -= bytesToRead;
  }

}

void setup() {

  Serial.begin(115200);
  
  if (setup_camera() != 0) {
    return;
  } 

  if (setup_bluetooth() != 0) {
    return;
  } 

}

void loop() {

  cam.reset();
  delay(300);
  
  // Wait for send command
  while (!btSerial.available() && btSerial.read() != '1') {}
  
  take_picture();
}

