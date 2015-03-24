#!/usr/local/bin/python

import serial
import io
import struct
import cStringIO
import time
import cv2
import numpy as np
import threading
import types

class UnoBuffer (threading.Thread):

    def __init__(self, buf):
        super(UnoBuffer, self).__init__()
        self.buf = buf
        self.hackberry = HackberrySerial()
        self._stop = threading.Event()

    def run(self):
        self.hackberry.connect()
        self.hackberry.flush()
        while not self.stopped():
            img = self.hackberry.get_frame()
            if img.size != 0 or type(img) == types.NoneType:
                self.buf.append(img)
            self.hackberry.flush()

        self.hackberry.disconnect()

    def stop(self):
        print 'UnoBuffer is stopped'
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

class HackberrySerial:

    def __init__(self):
        self.counter    = 0
        self.baud_rate  = 57600
        self.port_name  = '/dev/tty.HC-06-DevB'
        self.buff_size  = 64
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

        return filename

    def flush_input(self):
        self.ser.flushInput()

    def acceptable_size(self, size):
        return (size < 14000 and size > 10000)

    def read_frame_size(self):
        
        size_in_bytes = 14001
        
        while not self.acceptable_size(size_in_bytes):
            self.send_take_picture_cmd()
            size_in_bytes = self.ser.read(self.frame_size)
            size_in_bytes = int(size_in_bytes.encode('hex'), 16)

        return size_in_bytes

    def send_take_picture_cmd(self):
        print "\t--> Sent take picture command"
        self.ser.write('1')

    def read_frame(self, stream):

        buffer_read_avg = dict()

        buffer_read_avg['sum'] = 0
        buffer_read_avg['num'] = 0
        buffer_read_avg['avg'] = 0



        jpg_len = self.read_frame_size()

        if (jpg_len == 0):
            print "!!!! ERROR n frame: image len is 0 !!!!"
            return None

        frame_size = jpg_len
        print "\t--> New Frame of size = "+str(jpg_len)
        total_bytes_read = 0

        shit = False
        t_read = 0
        while frame_size > 0:
            bytes_to_read = min(self.buff_size, frame_size)

            ts = time.time()
            buf = self.ser.read(bytes_to_read)
            te = time.time() - ts
            t_read += len(buf)
            buffer_read_avg['sum'] += te
            buffer_read_avg['num'] += 1
            stream.write(buf)
            bytes_read = len(buf) 
            frame_size -= bytes_read
            total_bytes_read += bytes_read
            if jpg_len == total_bytes_read or shit:
                print '.'
                if buf[-2:].encode('hex') == "ffd9" or shit:
                    print "\t- Successful frame"
                    buffer_read_avg['avg'] = buffer_read_avg['sum']/buffer_read_avg['num']
                    print '\t- Average time for reading '+\
                            str(self.buff_size)+' = '+str(buffer_read_avg['avg'])
                    return True
                else:
                    # TODO: signal microController to resend
                    print buf[-2:].encode('hex')
                    print buf.encode('hex')
                    print "!!!! ERROR n frame, read "+str(t_read)+" !!!!"
                    return False

    def get_frame(self):

        stream = cStringIO.StringIO()
        ts = time.time()
        te = 0
        img = np.zeros([0,0,0], np.uint8)
        if self.read_frame(stream):
            te = time.time() - ts
            filename = self.stream_to_file(stream)
            img = cv2.imread(filename)
        stream.close()
        print '\t- Time to transfer an image = '+str(te)
        return img

    def flush(self):
        self.ser.flush()


    
if __name__ == '__main__':

  print 'hello, this is hackberry'

  hackberry = HackberrySerial()

  try: 
    hackberry.connect()
    hackberry.flush()
    while True:
        hackberry.get_frame()
        hackberry.flush()

  except KeyboardInterrupt:
    hackberry.disconnect()

