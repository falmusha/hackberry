#!/usr/bin/python

import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

HESSIAN_THRESHHOLD = 400

def run_on_path(path, surf):
  for dirname, dirs, filenames in os.walk(path):
    for filename in filenames:
      img_path = os.path.join(dirname, filename)
      img = cv2.imread(img_path)
      kp, des = surf.detectAndCompute(img, None)
      drawnImg = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)
      plt.imshow(drawnImg),plt.show()

if __name__ == "__main__":

  surf = cv2.SURF(HESSIAN_THRESHHOLD)
  
  test_path = sys.argv[1]
  if (test_path):
    run_on_path(test_path, surf)
  else:
    capture = cv2.VideoCapture(0)

    while(1):
      ret, frame = capture.read()

      kp, des = surf.detectAndCompute(frame, None)
      img = cv2.drawKeypoints(frame, kp, None, (255,0,0), 4)

      cv2.imshow('SURF', img)

      if cv2.waitKey(1) & 0xFF == ord('q'):
        img = frame
        break

    cv2.destroyAllWindows() 
