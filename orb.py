#!/usr/bin/python

import sys
import cv2
import numpy as np

if __name__ == "__main__":

  capture = cv2.VideoCapture(0)
  orb = cv2.ORB()

  while(1):
    ret, frame = capture.read()

    kp = orb.detect(frame, None)
    kp, des = orb.compute(frame, kp)
    img = cv2.drawKeypoints(frame, kp, color=(0,255,0), flags=0)
    cv2.imshow('ORB', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      img = frame
      break

  cv2.destroyAllWindows() 
