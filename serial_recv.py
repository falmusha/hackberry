#!/usr/bin/python

import serial
import io
import struct
import cStringIO


BAUD_RATE = 9600
PORT_NAME = '/dev/tty.HC-06-DevB'
BUFF_SIZE = 32
FRAME_SIZE = 2

counter = 0
def stream_to_file(stream):

  global counter
  counter += 1
  filename = "IMAGE_"+str(counter)+".jpg"

  with open(filename, 'wb') as f:
    f.write(stream.getvalue())

def serial_read_frame_size(ser_conn):

  # Wait until port has data to read
  while ser.inWaiting() <= 0:
    pass

  b = ser_conn.read(FRAME_SIZE)
  return int(b.encode('hex'), 16)

if __name__ == "__main__":

  # Hook serial port to bluetooth port
  ser = serial.Serial(PORT_NAME, BAUD_RATE, timeout=None)

  
  while True:
    try:

      f = cStringIO.StringIO()
      ser.write('1');
      jpg_len = serial_read_frame_size(ser)
      if (jpg_len == 0):
        f.close()
        continue

      frame_size = jpg_len
      print "--> New Frame of size = "+str(jpg_len)
      total_bytes_read = 0

      while frame_size > 0:
        bytes_to_read = min(BUFF_SIZE, frame_size)
        buf = ser.read(bytes_to_read)
        f.write(buf)
        bytes_read = len(buf) 
        frame_size -= bytes_read
        total_bytes_read += bytes_read
        if jpg_len == total_bytes_read:
          if buf[-2:].encode('hex') == "ffd9":
            print "Successful frame"
          else:
            print buf[-2:].encode('hex')
            print buf.encode('hex')
            print "!!!! ERROR n frame !!!!"
    
      stream_to_file(f)
      f.close()
    except KeyboardInterrupt:
      ser.close()
      f.close()

  #ser.close()
  #ser.flush()

  #f = open('pic.jpg', 'w')
  #c = open('size', 'w')

  #ser.write('1');
  
  #while ser.inWaiting() <= 0:
    #pass

  #count = 0
  #wrote_size = False

  #try:
    #while True:
      
      ##frame_length = 0
      ##while ser.inWaiting() > 0 and not wrote_size:
        ##buf = ser.read(FRAME_SIZE)
        ##c.write(buf)
        ##break
        

      ##raise KeyboardInterrupt

      #while ser.inWaiting() > 0:
        #buf = ser.read(BUFF_SIZE)
        #f.write(buf)
        #count += len(buf)

  #except KeyboardInterrupt:
    #print '\nClosing serial port and file'
    #f.close()
    #c.close()
    #ser.close()
    #print str(count)

