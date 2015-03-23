#!/usr/local/bin/python

import cv2
import numpy as np

import time
import pdb
import math


class Matcher:

    def __init__(self):
        # default FLANN parameters
        self.flann_matcher = cv2.FlannBasedMatcher(
                dict(algorithm=0, trees=5),
                dict(checks=50)
        )
        self.binary_flann_matcher = cv2.FlannBasedMatcher(
                dict(
                    algorithm=6,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1
                ), 
                dict(checks=50)
        )

        # default BF parameters
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        self.binary_bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # default KNN BF parameters
        self.knn_bf_matcher = cv2.BFMatcher()
        self.binary_knn_bf_matcher = cv2.BFMatcher()

    def set_flann_matcher(self, index_params, search_params):
        self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        return self.flann_matcher

    def set_binary_flann_match_params(self, index_params, search_params):
        self.binary_flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
        return self.binary_flann_matcher

    def filter_matches(self, matches, ratio = 0.7):
        filtered = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                filtered.append(m[0])
        
        return filtered

    def brute_force_match(self, des1, des2):
        return self.bf_matcher.match(des1, des2)

    def binary_brute_force_match(self, des1, des2):
        return self.binary_bf_matcher.match(des1, des2)

    def knn_brute_force_match(self, des1, des2, k=2):
        matches = self.knn_bf_matcher.knnMatch(des1, des2, k=k)
        return self.filter_matches(matches)

    def binary_knn_brute_force_match(self, des1, des2, k=2):
        matches = self.binary_knn_bf_matcher.knnMatch(des1, des2, k=k)
        return self.filter_matches(matches)

    def flann_match(self, des1, des2, k=2):
        matches = self.flann_matcher.knnMatch(des1, des2, k=k)
        return self.filter_matches(matches)

    def binary_flann_match(self, des1, des2, k=2):
        matches = self.binary_flann_matcher.knnMatch(des1, des2, k=k)
        return self.filter_matches(matches)


class ComputerVision:

    def __init__(self):
        self.matcher = Matcher()

    def show(self, name, img):
        cv2.imshow(name, img)
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows() 

    def draw_matches(self, image1, image2, points1, points2):
        'Connects corresponding features in the two images using yellow lines.'

        ## Put images side-by-side into 'image'.
        offset = 0
        (h1, w1) = image1.shape[:2]
        (h2, w2) = image2.shape[:2]

        if len(image1.shape) > 2:
            image = np.zeros((max(h1, h2), w1 + w2 + offset, 3), np.uint8)
        else:
            image = np.zeros((max(h1, h2), w1 + w2 + offset), np.uint8)

        image[:h1, :w1] = image1
        image[:h2, w1+offset:w1+offset+w2] = image2
        
        ## Draw yellow lines connecting corresponding features.
        for (x1, y1), (x2, y2) in zip(np.int32(points1), np.int32(points2)):
            cv2.line(image, (x1, y1), (x2+w1, y2), (0, 255, 255), lineType=cv2.CV_AA)

        return image

    def find_dimensions(self, img_size, homography):

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

            h_pt = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T

            h_pt_arr = np.array(h_pt, np.float32)

            c_pt = np.array([h_pt_arr[0]/h_pt_arr[2], h_pt_arr[1]/h_pt_arr[2]], np.float32)

            if ( max_x == None or c_pt[0,0] > max_x ):
                max_x = c_pt[0,0]
            if ( max_y == None or c_pt[1,0] > max_y ):
                max_y = c_pt[1,0]
            if ( min_x == None or c_pt[0,0] < min_x ):
                min_x = c_pt[0,0]
            if ( min_y == None or c_pt[1,0] < min_y ):
                min_y = c_pt[1,0]

        min_x = min(0, min_x)
        min_y = min(0, min_y)

        return (min_x, min_y, max_x, max_y)

    def calc_size(self, size_image1, size_image2, homography):
    
        (min_x, min_y, max_x, max_y) = self.find_dimensions(size_image2, homography)

        min_x = int(math.ceil(min_x))
        min_y = int(math.ceil(min_y))
        max_x = int(math.ceil(max_x))
        max_y = int(math.ceil(max_y))

        max_x = max(size_image1[0], max_x)
        max_y = max(size_image1[1], max_y)

        x_offset = 0
        y_offset = 0

        if min_x < 0:
            x_offset += -(min_x)
            max_x += -(min_x)
        if min_y < 0:
            y_offset += -(min_y)
            max_y += -(min_y)


        offset = (x_offset, y_offset)
        size   = (max_y, max_x)
        
        homography[0:2,2] +=  offset

        return (size, offset)

    def crop_off_black_edges(self, final_img):

        # Crop off the black edges
        final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        max_area = 0
        best_rect = (0,0,0,0)

        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            # print "Bounding Rectangle: ", (x,y,w,h)

            deltaHeight = h-y
            deltaWidth = w-x

            area = deltaHeight * deltaWidth

            if ( area > max_area and deltaHeight > 0 and deltaWidth > 0):
                max_area = area
                best_rect = (x,y,w,h)

        if ( max_area > 0 ):
            final_img_crop = final_img[best_rect[1]:best_rect[1]+best_rect[3],
                    best_rect[0]:best_rect[0]+best_rect[2]]
            return final_img_crop
        else:
            return final_img

    def detect_and_compute(self, frame, kp_a, des_a):
        ''' kp_a is keypoint algorithim,
            des_a is decription algorithim
        '''
        kp = kp_a.detect(frame, None)
        return des_a.compute(frame, kp)

    def find_homography(self, src_pts, dst_pts):
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        outlier_indices = []
        for i,m in enumerate(mask):
            if m[0] == 0:
                outlier_indices.append(i)

        return (H, outlier_indices)

    def stitch(self, img1, img2, kp_alg, des_alg, min_match=4):

        kp1, des1 = self.detect_and_compute(img1, kp_alg, des_alg)
        kp2, des2 = self.detect_and_compute(img2, kp_alg, des_alg)

        matches = self.matcher.flann_match(des1, des2)

        if len(matches) < min_match:
            print "Not enough matches are found - %d/%d" % (len(matches), min_match)
            raise Exception

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])

        #drawn_matches = self.draw_matches(img1, img2, src_pts, dst_pts)
        #self.show('matches', drawn_matches)

        (H, outlier_indices) = self.find_homography(dst_pts, src_pts)

        #src_pts = np.delete(src_pts, outlier_indices, 0)
        #dst_pts = np.delete(dst_pts, outlier_indices, 0)
        #drawn_matches = self.draw_matches(img1, img2, src_pts, dst_pts)
        #self.show('new matches', drawn_matches)

        (o_size, offset) = self.calc_size(img1.shape, img2.shape, H)

        dst_h = o_size[0] # y
        dst_w = o_size[1] # x

        offset_h = np.matrix(np.identity(3), np.float32)
        offset_h[0,2] = offset[0]
        offset_h[1,2] = offset[1]

        warped_1 = cv2.warpPerspective(
                    img1,
                    offset_h,
                    (dst_w, dst_h)
                )
        #self.show('w1', warped_1)

        warped_2 = cv2.warpPerspective(
                    img2,
                    H,
                    (dst_w, dst_h)
                )
        #self.show('w2', warped_2)


        warped_2_gray = cv2.cvtColor(warped_2, cv2.COLOR_BGR2GRAY)
        (_, warped_2_mask) = cv2.threshold(warped_2_gray, 0, 255, cv2.THRESH_BINARY)

        out = np.zeros((dst_h, dst_w, 3), np.uint8)
        out = cv2.add(out, warped_1,
                mask=np.bitwise_not(warped_2_mask),
                dtype=cv2.CV_8U)
        out = cv2.add(out, warped_2, dtype=cv2.CV_8U)
        out = self.crop_off_black_edges(out)

        #self.show('o', small_out)

        return out
