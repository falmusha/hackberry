#!/usr/bin/python

import sys
import cv2
import numpy as np

if __name__ == "__main__":

  capture = cv2.VideoCapture(0)
  fast = cv2.FastFeatureDetector()

  while(1):
    ret, frame = capture.read()

    kp = fast.detect(frame, None)
    img = cv2.drawKeypoints(frame, kp, color=(255,0,0))

    cv2.imshow('FAST', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      img = frame
      break

  cv2.destroyAllWindows() 
