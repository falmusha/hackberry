#!/usr/bin/python

import cStringIO
from hackberry_serial import HackberrySerial
from hackberry_algorithms import *

if __name__ == '__main__':

  print 'hello, this is hackberry'

  hackberry_ser = HackberrySerial()
  hackberry_ser.connect()

  hackberry_cv = HackberryCV()

  stream = cStringIO.StringIO()
  hackberry_ser.read_frame(stream)
  img = hackberry_cv.surf_keypoints(stream)
  stream.close()

  cv2.imshow('SURF', img)

  try: 
    while True:
      key = cv2.waitKey(33)
      if key == 27:
        break
      elif key == 32:
        s = cStringIO.StringIO()
        hackberry_ser.read_frame(s)
        img = hackberry_cv.surf_keypoints(s)
        s.close()
        cv2.imshow('SURF', img)



  except KeyboardInterrupt:
    hackberry_ser.disconnect()
    stream.close()
