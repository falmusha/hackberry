#!/usr/bin/python

import sys
import cv2
import numpy as np


if __name__ == "__main__":

  capture = cv2.VideoCapture(0)
  sift = cv2.SIFT()

  while(1):
    ret, frame = capture.read()

    kp = sift.detect(frame, None)
    #img = cv2.drawKeypoints(frame, kp)
    img = cv2.drawKeypoints(frame, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    

    cv2.imshow('CLIENT', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      img = frame
      break

  cv2.destroyAllWindows() 
