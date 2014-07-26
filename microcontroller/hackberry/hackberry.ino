#include <Adafruit_VC0706.h>
#include <SD.h>
#include <SPI.h>
#include <SoftwareSerial.h>         
#include <AltSoftSerial.h>

// Adafruit SD shields and modules: pin 10
#define chipSelect 10

#define CAMERA_TX 2
#define CAMERA_RX 3
#define BLUETOOTH_TX 8
#define BLUETOOTH_RX 9
#define BLUETOOTH_STATE 4
#define BAUD_RATE 9600
#define LED_PIN 7
#define BUF_SIZE 64
// Using SoftwareSerial (Arduino 1.0+) or NewSoftSerial (Arduino 0023 & prior):
#if ARDUINO >= 100
// On Uno: camera TX connected to pin 2, camera RX to pin 3:
SoftwareSerial camera_connection = SoftwareSerial(CAMERA_TX, CAMERA_RX);
#else
NewSoftSerial camera_connection = NewSoftSerial(CAMERA_TX, CAMERA_RX);
#endif

AltSoftSerial btSerial = AltSoftSerial(BLUETOOTH_TX, BLUETOOTH_RX);

Adafruit_VC0706 cam = Adafruit_VC0706(&camera_connection);

char buf[BUF_SIZE];

int setup_sd(void) {

  // see if the card is present and can be initialized:
  if (SD.begin(chipSelect)) {
    Serial.println("Card is present");
    return 0;
  } else {
    Serial.println("Failed, Card Not Found");
    return -1;
  } 
}

int setup_camera(void) {

  // Try to locate the camera
  if (cam.begin()) {
    Serial.println("Camera Found");
    return 0;
  } else {
    Serial.println("Failed, Camera Not Found");
    return -1;
  } 

  /* Choose camera setting */
  //cam.setImageSize(VC0706_640x480);        // biggest
  /*cam.setImageSize(VC0706_320x240);        // medium*/
  cam.setImageSize(VC0706_160x120);          // small

}

int setup_bluetooth() {

  // Bluetooth initialization:
  btSerial.begin(BAUD_RATE);
  pinMode(LED_PIN, OUTPUT);
  pinMode(BLUETOOTH_STATE, INPUT);
  delay(300);
  if (btSerial.isListening()) {
    Serial.println("Bluetooth Found");
    return 0;
  } else {
    Serial.println("Failed, Bluetooth Not Found");
    return -1;
  }

}

void take_picture() {

  if (!cam.takePicture()) {
    Serial.println("Failed to snap!");
  }

  // Get the size of the image (frame) taken  
  uint16_t jpglen = cam.frameLength();
  Serial.print("Storing ");
  Serial.print(jpglen, DEC);
  Serial.print(" byte image.");

  int32_t time = millis();

  // Read all the data up to # bytes!
  // For counting # of writes
  byte wCount = 0; 

  while (jpglen > 0) {

    // read 32 bytes at a time;
    uint8_t *buffer;

    // change 32 to 64 for a speedup but may not work with all setups!
    uint8_t bytesToRead = min(32, jpglen); 
    buffer = cam.readPicture(bytesToRead);
    delay(1);
    btSerial.write(buffer, bytesToRead);
    delay(30);

    // Every 2K, give a little feedback so it doesn't appear locked up
    if(++wCount >= 64) { 
      wCount = 0;
    }

    jpglen -= bytesToRead;
  }

  time = millis() - time;

  Serial.println("done!");

  Serial.print(time); 
  
  Serial.println(" ms elapsed");

}

void setup() {

  int err = 0;
// When using hardware SPI, the SS pin MUST be set to an
// output (even if not connected or used).  If left as a
// floating input w/SPI on, this can cause lockuppage.
#if !defined(SOFTWARE_SPI)
  if(chipSelect != 10) pinMode(10, OUTPUT); // SS on Uno, etc.
#endif

  Serial.begin(BAUD_RATE);
  Serial.println("-------[ Hackberry ]-------");
  
  /*if ((err = setup_sd()) != 0) {*/
    /*return;*/
  /*} */
  
  if ((err = setup_camera()) != 0) {
    return;
  } 

  if ((err = setup_bluetooth()) != 0) {
    return;
  } 

  while(!Serial) {}

  delay(300);

  take_picture();
}

void loop() {
}

