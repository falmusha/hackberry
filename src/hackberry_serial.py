#!/usr/bin/python

import serial
import io
import struct
import cStringIO

class HackberrySerial:

  def __init__(self):
    self.counter    = 0
    self.baud_rate  = 9600
    self.port_name  = '/dev/tty.HC-06-DevB'
    self.buff_size  = 32
    self.frame_size = 2
    self.ser        = None

  def connect(self):
    # Hook serial port to bluetooth port
    self.ser = serial.Serial(self.port_name, self.baud_rate, timeout=None)

  def disconnect(self):
    # Hook serial port to bluetooth port
    self.ser.close()

  def stream_to_file(self, stream):

    self.counter += 1
    filename = "IMAGE_"+str(self.counter)+".jpg"
    
    print 'Writing image to file '+filename

    with open(filename, 'wb') as f:
      f.write(stream.getvalue())

  def read_frame_size(self):

    # Wait until port has data to read
    while self.ser.inWaiting() <= 0:
      pass

    size_in_bytes = self.ser.read(self.frame_size)
    return int(size_in_bytes.encode('hex'), 16)

  def send_take_picture_cmd(self):
      self.ser.write('1')
  
  def read_frame(self, stream):

    self.send_take_picture_cmd()

    jpg_len = self.read_frame_size()

    if (jpg_len == 0):
      print "!!!! ERROR n frame: image len is 0 !!!!"
      return None

    frame_size = jpg_len
    print "--> New Frame of size = "+str(jpg_len)
    total_bytes_read = 0

    while frame_size > 0:
      bytes_to_read = min(self.buff_size, frame_size)

      while self.ser.inWaiting() < bytes_to_read:
        pass

      buf = self.ser.read(bytes_to_read)
      stream.write(buf)
      bytes_read = len(buf) 
      frame_size -= bytes_read
      total_bytes_read += bytes_read
      if jpg_len == total_bytes_read:
        if buf[-2:].encode('hex') == "ffd9":
          print "Successful frame"
          return True
        else:
          # TODO: signal microController to resend
          print buf[-2:].encode('hex')
          print buf.encode('hex')
          print "!!!! ERROR n frame !!!!"
          return False

  def get_frame(self):

    stream = cStringIO.StringIO()
    if self.read_frame(stream):
      self.stream_to_file(stream)
    stream.close()

    
if __name__ == '__main__':

  print 'hello, this is hackberry'

  hackberry = HackberrySerial()

  try: 
    while True:
      hackberry.connect()
      hackberry.get_frame()

  except KeyboardInterrupt:
    hackberry.disconnect()

