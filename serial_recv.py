#!/usr/bin/python

import serial
import io

BAUD_RATE = 9600
PORT_NAME = '/dev/tty.HC-06-DevB'
BUFF_SIZE = 32

if __name__ == "__main__":

  # Hook serial port to bluetooth port
  ser = serial.Serial(PORT_NAME, BAUD_RATE, timeout=1)

  # Flush port
  ser.flush()

  f = open('pic.jpg', 'wb')

  try:
    while True:
      while ser.inWaiting() > 0:
        f.write(ser.read())
  except KeyboardInterrupt:
    print 'Closing serial port and file'
    f.close()
    ser.close()

