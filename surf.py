#!/usr/bin/python

import sys
import cv2
import numpy as np

HESSIAN_THRESHHOLD = 400

if __name__ == "__main__":

  capture = cv2.VideoCapture(0)
  surf = cv2.SURF(HESSIAN_THRESHHOLD)

  while(1):
    ret, frame = capture.read()

    kp, des = surf.detectAndCompute(frame, None)
    img = cv2.drawKeypoints(frame, kp, None, (255,0,0), 4)

    cv2.imshow('SURF', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      img = frame
      break

  cv2.destroyAllWindows() 
