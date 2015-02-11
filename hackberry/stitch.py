#!/usr/bin/python

import pdb
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

def get_matcher():

    FLANN_INDEX_KDTREE = 0

    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    return flann

def stitch(imgs):

    pdb.set_trace()

    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv2.SIFT()


    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(imgs[0],None)
    kp2, des2 = sift.detectAndCompute(imgs[1],None)

    flann_matcher = get_matcher()
    matches = flann_matcher.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        pdb.set_trace()

        img1_rows  = imgs[0].shape[0]
        img1_colms = imgs[0].shape[1]
        img2_rows  = imgs[1].shape[0]
        img2_colms = imgs[1].shape[1]

        warped_1 = cv2.warpPerspective(
                    imgs[0],
                    M, 
                    (img1_colms+img2_colms, img1_rows)
                )

        pdb.set_trace()

    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None


if __name__ == "__main__":

    if sys.argv < 2:
        exit(0)

    img1 = cv2.imread(sys.argv[1],0) # queryImage
    img2 = cv2.imread(sys.argv[2],0) # trainImage

    stitched_img = stitch([img1, img2])

    cv2.imshow('stitched', stitched_img)
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows() 
