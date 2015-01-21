#!/usr/bin/python

import numpy as np
import cv2
import fast
import surf

class ComputerVision:

  def __init__(self):
    self.surf = None
    self.fast = None

  def init_surf(self):
    if not self.surf:
      self.surf = Surf()

  def init_fast(self):
    if not self.fast:
      self.fast = Fast()

  def stream_to_np_array(self, stream):
    img_string = stream.getvalue()
    np_array = np.fromstring(img_string, dtype=np.uint8)
    return cv2.imdecode(np_array, 0)

  def surf_keypoints(self, frame):

    if not self.surf:
      print "Not init"

    return self.surf.draw_keypoints(frame)

  def fast_keypoints(self, frame):

    if not self.fast:
      print "Not init"

    return self.fast.draw_keypoints(frame)
