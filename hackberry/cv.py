#!/usr/bin/python

import cv2

class ComputerVision:

    def __init__(self):
        self.x = None

    def detect_and_compute(self, frame, kp_a, des_a):
        ''' kp_a is keypoint algorithim,
            des_a is decription algorithim
        '''
        kp = kp_a.detect(frame, None)
        return des_a.compute(frame, kp)
        


if __name__ == "__main__":
    pass

