#!/usr/bin/python

import cStringIO
from hackberry_serial import HackberrySerial
from hackberry_algorithms import *

if __name__ == '__main__':

  print 'hello, this is hackberry'

  hackberry_ser = HackberrySerial()
  hackberry_ser.connect()

  hackberry_cv = HackberryCV()
  hackberry_cv.init_surf()
  hackberry_cv.init_fast()


  stream = cStringIO.StringIO()
  hackberry_ser.read_frame(stream)
  frame = hackberry_cv.stream_to_np_array(stream)
  img = hackberry_cv.surf_keypoints(frame)
  img2 = hackberry_cv.fast_keypoints(frame)
  stream.close()

  cv2.imshow('SURF', img)
  cv2.imshow('FAST', img2)

  try: 
    while True:
      key = cv2.waitKey(33)
      if key == 27:
        break
      elif key == 32:
        print "++++ Take new picture"
        s = cStringIO.StringIO()
        hackberry_ser.read_frame(s)
        frame = hackberry_cv.stream_to_np_array(s)
        img = hackberry_cv.surf_keypoints(frame)
        img2 = hackberry_cv.fast_keypoints(frame)
        s.close()
        cv2.imshow('SURF', img)
        cv2.imshow('FAST', img2)



  except KeyboardInterrupt:
    hackberry_ser.disconnect()
    stream.close()
