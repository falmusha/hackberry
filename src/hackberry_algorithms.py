#!/usr/bin/python

import numpy as np
import cv2

class HackberryCV:

  def __init__(self):
    self.surf = None

  def init_surf(self):
    if not self.surf:
      self.surf = HackberrySurfCV()

  def stream_to_np_array(self, stream):
    img_string = stream.getvalue()
    np_array = np.fromstring(img_string, dtype=np.uint8)
    return cv2.imdecode(np_array, 0)

  def surf_keypoints(self, stream):

    if not self.surf:
      self.init_surf()

    frame = self.stream_to_np_array(stream)
    return self.surf.draw_keypoints(frame)


class HackberrySurfCV:

  def __init__(self):
    self.hessian_threshhold = 400
    self.surf = cv2.SURF(self.hessian_threshhold)
    self.window_name = 'SURF'

  def destroy(self):
    cv2.destroyAllWindows() 

  def draw_keypoints(self, frame):

    kp, des = self.surf.detectAndCompute(frame, None)
    img = cv2.drawKeypoints(frame, kp, None, (255,0,0), 4)

    return img

