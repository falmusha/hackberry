#!/usr/bin/python

import pdb
import os
import shutil
import sys
import csv
import cv2
import time
import n_cv

class HackberryBenchmark:

    def __init__(self, path, algorithims=[], image_ext=['jpg']):
        self.hcv = n_cv.ComputerVision()
        self.image_ext = image_ext
        self.path = path
        self.algorithims = algorithims

    def get_cv_algorithims(self, types=False):
        return self.algorithims

    def get_files(self):
        files = []
        for f in os.listdir(self.path):
            ext = os.path.splitext(f)[1][1:].lower()
            if ext in self.image_ext:
                files.append(os.path.join(self.path, f)) 
        return files

    def get_matchers(self):
        pass

    def generate_matchers(self, f):
        pass

    def generate_keypoints(self, f):
        ''' Generate kps for all files in path and write them in
            f as csv file
        '''

        csvwriter = csv.writer(f)

        csv_header = ['Algorithim', 'Image', 'Time', 'Keypoints']
        csvwriter.writerow(csv_header)

        for a in self.get_cv_algorithims():

            for img in self.get_files():

                row = []
                row.append(a.__class__.__name__)
                row.append(os.path.basename(img))
                frame = cv2.imread(img)

                t_start = time.time()
                kp = a.detect(frame, None)
                t_end = round(time.time() - t_start, 3)

                row.append(t_end)
                row.append(len(kp))

                csvwriter.writerow(row)
                self.write_img(img, frame, kp, a.__class__.__name__)

        return None
        
    def write_img(self, img_file, frame, kp, a_name):
        

        img_name, img_ext = os.path.splitext(img_file)

        img_name = os.path.basename(img_name)

        img_name = img_name + '_' + a_name + img_ext

        img = cv2.drawKeypoints(frame, kp, None, (255,0,0), 4)

        kp_path = os.path.join(self.path, 'kp')

        if not os.path.isdir(kp_path):
            os.mkdir(kp_path)

        cv2.imwrite(os.path.join(kp_path, img_name), img)
      
        
if __name__ == "__main__":
    pass

    if sys.argv < 2:
        exit(0)

    path = sys.argv[1]
    algorithims = [
                cv2.SIFT(),
                cv2.SURF(),
                cv2.FastFeatureDetector(),
                cv2.ORB(),
            ]

    benchmark = HackberryBenchmark(path, algorithims, ['jpg'])

    with open('keypoints.csv', 'w+') as f:
        benchmark.generate_keypoints(f)

    with open('descriptions.csv', 'w+') as f:
        benchmark.generate_keypoints(f)


