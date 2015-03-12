#!/usr/bin/python

import pdb
import sys
import os
import cv2
import numpy as np
import math
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


def find_dimensions(img_size, homography):

    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    (y, x) = img_size[:2]

    base_p1[:2] = [0,0]
    base_p2[:2] = [x,0]
    base_p3[:2] = [0,y]
    base_p4[:2] = [x,y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None
    for pt in [base_p1, base_p2, base_p3, base_p4]:

        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

        hp_arr = np.array(hp, np.float32)

        normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)

        if ( max_x == None or normal_pt[0,0] > max_x ):
            max_x = normal_pt[0,0]

        if ( max_y == None or normal_pt[1,0] > max_y ):
            max_y = normal_pt[1,0]

        if ( min_x == None or normal_pt[0,0] < min_x ):
            min_x = normal_pt[0,0]

        if ( min_y == None or normal_pt[1,0] < min_y ):
            min_y = normal_pt[1,0]
    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return (min_x, min_y, max_x, max_y)

def calc_size(size_image1, size_image2, homography):
  
    (min_x, min_y, max_x, max_y) = find_dimensions(size_image2, homography)

    min_x = int(math.ceil(min_x))
    min_y = int(math.ceil(min_y))
    max_x = int(math.ceil(max_x))
    max_y = int(math.ceil(max_y))

    x_offset = 0
    y_offset = 0

    if min_x < 0:
        x_offset += -(min_x)
    if min_y < 0:
        y_offset += -(min_y)

    max_x = int(math.ceil(max(size_image1[0], max_x)))
    max_y = int(math.ceil(max(size_image1[1], max_y)))

    offset = (x_offset, y_offset)
    size   = (max_y, max_x)
    
    homography[0:2,2] +=  offset

    #sizes = {
            #'out': size,
            #'1': size_image1,
            #'2': img2_dims,
            #}

    return (size, offset)

def stitch_arr(imgs):

    imgs_len = len(imgs)

    if imgs_len == 0:
        return None

    out = imgs[0]
    for i in range(0, imgs_len):
        if i+1 == imgs_len:
            return out
        out = stitch(out, imgs[i+1])

def stitch(img1, img2):

    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv2.SIFT()


    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    flann_matcher = get_matcher()
    matches = flann_matcher.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        (o_size, offset) = calc_size(img1.shape, img2.shape, H)
        (o_size, offset) = calc_size(img1.shape, img2.shape, H)

        # y
        dst_h = o_size[0]
        # x
        dst_w = o_size[1]

        offset_h = np.matrix(np.identity(3), np.float32)
        offset_h[0,2] = offset[0]
        offset_h[1,2] = offset[1]

        warped_1 = cv2.warpPerspective(
                    img1,
                    offset_h,
                    (dst_w, dst_h)
                )

        warped_2 = cv2.warpPerspective(
                    img2,
                    H,
                    (dst_w, dst_h)
                )

        out = np.zeros((dst_h, dst_w), np.uint8)
        out = cv2.add(out, warped_1, dtype=cv2.CV_8U)
        out = cv2.add(out, warped_2, dtype=cv2.CV_8U)

        return out
    else:
        print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
        matchesMask = None


if __name__ == "__main__":

    if sys.argv < 2:
        exit(0)

    img1 = cv2.imread(sys.argv[1],0) # queryImage
    img2 = cv2.imread(sys.argv[2],0) # trainImage

    stitched_img = stitch_arr([img1, img2])

    cv2.imshow('stitched', stitched_img)
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows() 
