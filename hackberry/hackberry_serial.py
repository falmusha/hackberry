#!/usr/bin/python

import serial
import io
import struct
import cStringIO
import time

class HackberrySerial:

    def __init__(self):
        self.counter    = 0
        #self.baud_rate  = 9600
        #self.baud_rate  = 19200
        #self.baud_rate  = 38400
        self.baud_rate  = 57600
        #self.baud_rate  = 115200
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

    def read_frame_size(self):

        # Wait until port has data to read
        #while self.ser.inWaiting() < 2:
            #continue

        size_in_bytes = self.ser.read(self.frame_size)
        self.send_ack()
        return int(size_in_bytes.encode('hex'), 16)

    def send_take_picture_cmd(self):
        print "\t--> Sent take picture command"
        self.ser.write('1')

    def send_ack(self):
        self.ser.write('7')
    
    def read_frame(self, stream):

        buffer_read_avg = dict()

        buffer_read_avg['sum'] = 0
        buffer_read_avg['num'] = 0
        buffer_read_avg['avg'] = 0

        self.send_take_picture_cmd()


        #while self.ser.inWaiting() < 16:
            #continue

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

            #while self.ser.inWaiting() < bytes_to_read:
                #continue
            ts = time.time()
            buf = self.ser.read(bytes_to_read)
            te = time.time() - ts
            #self.send_ack()
            t_read += len(buf)
            #x = [b.encode('hex') for b in buf]
            #for i, b in enumerate(x): 
                #if (i+1) == len(x):
                    #break
                #if b == "ff":
                    #if x[i+1] == "d9":
                        #print 'shit'
                        #shit == True
                        #break
                        #import pdb;pdb.set_trace()
            #print '.',
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
        if self.read_frame(stream):
            te = time.time() - ts
            self.stream_to_file(stream)
        stream.close()
        print '\t- Time to transfer an image = '+str(te)

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

