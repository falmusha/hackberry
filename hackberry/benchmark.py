#!/usr/local/bin/python

import pdb
import os
import shutil
import sys
import csv
import cv2
import numpy as np
import time
import hcv

class HackberryBenchmark:

    def __init__(self, path, kp_a=[], des_a=[], image_ext=['jpg']):
        self.hcv = hcv.ComputerVision()
        self.h_matcher = hcv.Matcher()
        self.image_ext = image_ext
        self.path = path
        self.kp_a = kp_a
        self.des_a = des_a

    def get_cv_kp_algorithims(self, types=False):
        return self.kp_a

    def get_cv_des_algorithims(self, types=False):
        return self.des_a

    def get_files(self):
        files = []
        for f in os.listdir(self.path):
            ext = os.path.splitext(f)[1][1:].lower()
            if ext in self.image_ext:
                files.append(os.path.join(self.path, f)) 
        return files

    def get_algorithim_name(self, a):
        return a.__class__.__name__.lower()

    def write_match_results_to_file(self, img1, img2, kp1, kp2, matches, f_name):

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])

        if len(matches) >= 4:
            (_, outlier_indices) = self.hcv.find_homography(dst_pts, src_pts)

            src_pts = np.delete(src_pts, outlier_indices, 0)
            dst_pts = np.delete(dst_pts, outlier_indices, 0)

        match_img = self.hcv.draw_matches(img1, img2, src_pts, dst_pts)

        result_path = os.path.join(self.path, 'matches')
        if not os.path.isdir(result_path):
            os.mkdir(result_path)

        f_name = os.path.join(result_path, f_name+'.jpg')
        cv2.imwrite(f_name+'.jpg', match_img)

    def remove_outliers(self, kp1, kp2, matches):

        if len(matches) < 4:
            return (0,0)

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])

        (_, outlier_indices) = self.hcv.find_homography(dst_pts, src_pts)

        outliers = len(outlier_indices)

        src_pts = np.delete(src_pts, outlier_indices, 0)
        dst_pts = np.delete(dst_pts, outlier_indices, 0)

        inlier = abs(len(matches)-outliers)

        return (outliers, inlier)

    def get_match_results(self, kp1, kp2, des1, des2, descriptor_name):

        if descriptor_name == 'orb' or descriptor_name == 'brief':
            bf_r = dict()
            t = time.time()
            bf_r['matches'] = self.h_matcher.binary_brute_force_match(des1, des2) 
            bf_r['time'] = round(time.time()-t, 6)
            bf_r['outliers'], bf_r['inliers'] = self.remove_outliers(kp1, kp2, bf_r['matches'])

            
            knn_bf_r = dict()
            t = time.time()
            knn_bf_r['matches'] = self.h_matcher.binary_knn_brute_force_match(des1, des2) 
            knn_bf_r['time'] = round(time.time()-t, 6)
            knn_bf_r['outliers'], knn_bf_r['inliers'] = self.remove_outliers(kp1, kp2, knn_bf_r['matches'])

            flann_r = dict()
            t = time.time()
            flann_r['matches'] = self.h_matcher.binary_flann_match(des1, des2) 
            flann_r['time'] = round(time.time()-t, 6)
            flann_r['outliers'], flann_r['inliers'] = self.remove_outliers(kp1, kp2, flann_r['matches'])
        else:
            bf_r = dict()
            t = time.time()
            bf_r['matches'] = self.h_matcher.brute_force_match(des1, des2) 
            bf_r['time'] = round(time.time()-t, 6)
            bf_r['outliers'], bf_r['inliers'] = self.remove_outliers(kp1, kp2, bf_r['matches'])

            
            knn_bf_r = dict()
            t = time.time()
            knn_bf_r['matches'] = self.h_matcher.knn_brute_force_match(des1, des2) 
            knn_bf_r['time'] = round(time.time()-t, 6)
            knn_bf_r['outliers'], knn_bf_r['inliers'] = self.remove_outliers(kp1, kp2, knn_bf_r['matches'])

            flann_r = dict()
            t = time.time()
            flann_r['matches'] = self.h_matcher.flann_match(des1, des2) 
            flann_r['time'] = round(time.time()-t, 6)
            flann_r['outliers'], flann_r['inliers'] = self.remove_outliers(kp1, kp2, flann_r['matches'])

        return {
            'bf'     : bf_r,
            'knn_bf' : knn_bf_r, 
            'flann'  : flann_r, 
        }

    def detect_and_compute(self, img, feature_detector, descriptor):

        descriptor_name = descriptor[0]
        feature_detector_name = feature_detector[0]

        try:
            kp, des = self.hcv.detect_and_compute(img, feature_detector[1], descriptor[1])
        except cv2.error as e:
            kp = feature_detector[1].detect(img, None)
            des = np.empty([0,0])
            print('Something happened when computing '
                    + descriptor_name + '-'
                    + feature_detector_name + ' discriptions')

        return (kp, des)

    def generate_matches(self, img1, img2, draw=False):

        matches = dict()
        for feature_detector in self.get_cv_kp_algorithims():
            feature_detector_name = feature_detector[0]

            for descriptor in self.get_cv_des_algorithims():
                descriptor_name = descriptor[0]

                print('-----------------------------------')
                print('\t'+feature_detector_name+'-'+descriptor_name)
                print('-----------------------------------')
                kp1, des1 = self.detect_and_compute(img1, feature_detector, descriptor)
                kp2, des2 = self.detect_and_compute(img2, feature_detector, descriptor)
                
                if des1.size == 0 and des2.size == 0:
                    continue

                key = feature_detector_name+'-'+descriptor_name
                matches[key] = self.get_match_results(kp1, kp2, des1, des2, descriptor_name)

                for match in matches[key].keys():
                    _m = matches[key][match]['matches']
                    self.write_match_results_to_file(img1, img2, kp1, kp2, _m, key+'-'+match)

        return matches

    def generate_descriptions(self, kps):


        for kp_row in kps:

            img = os.path.join(self.path, kp_row['image'])
            frame = cv2.imread(img, 0)

            for descriptor in self.get_cv_des_algorithims():

                d_name = descriptor[0] + '_descriptions'
                des_name = descriptor[0] + '_d'

                try:
                    t_start = time.time()
                    des = descriptor[1].compute(frame, kp_row['kps'])[1]
                    t_end = round(time.time() - t_start, 6)
                except cv2.error as e:
                    des = np.empty([0,0])
                    kp_row['time_'+des_name] = 'na'
                    kp_row['time_per_'+des_name] = 'na'
                    kp_row[d_name] = 'na'
                    kp_row[des_name] = des
                    print('Something happened when computing '
                            + kp_row['algorithim'] + '/'
                            + d_name + ' discriptions for ' + img)
                    continue

                kp_row['time_'+des_name] = t_end
                kp_row['time_per_'+des_name] = t_end/des.shape[0]
                kp_row[d_name] = str(des.shape)
                kp_row[des_name] = des

        return kps

    def generate_keypoints(self):

        rows = []

        for a in self.get_cv_kp_algorithims():

            for img in self.get_files():

                row = {}
                row['algorithim'] = a[0]
                row['image'] = os.path.basename(img)
                frame = cv2.imread(img, 0)

                t_start = time.time()
                kp = a[1].detect(frame, None)
                t_end = round(time.time() - t_start, 6)

                row['kp_time'] = t_end
                row['time_per_keypoint'] = t_end/len(kp)
                row['keypoints'] = len(kp)
                row['kps'] = kp

                self.write_img(img, frame, kp, a[0])

                rows.append(row)

        return rows
        
    def write_img(self, img_file, frame, kp, a_name):
        

        img_name, img_ext = os.path.splitext(img_file)

        img_name = os.path.basename(img_name)

        img_name = img_name + '_' + a_name + img_ext

        img = cv2.drawKeypoints(frame, kp, None, (255,0,0), 4)

        kp_path = os.path.join(self.path, 'keypoints')

        if not os.path.isdir(kp_path):
            os.mkdir(kp_path)

        cv2.imwrite(os.path.join(kp_path, img_name), img)
      
    
    def csv_match_stats(self, img1, img2):

        matches = self.generate_matches(img1, img2)

        with open('matcher_stats.csv', 'w+') as f:
            csvwriter = csv.writer(f)

            csv_header = [
                'FeatureExtractor/Descriptor', 
                'BF Matches', 
                'BF Outlier', 
                'BF Inliers', 
                'BF Time', 
                'KNN BF Matches', 
                'KNN BF Outlier', 
                'KNN BF Inlier', 
                'KNN BF Time', 
                'FLANN Matches', 
                'FLANN Outlier', 
                'FLANN Inlier', 
                'FLANN Time', 
            ]

            csvwriter.writerow(csv_header)
            for m in matches:
                
                csvwriter.writerow([
                    m,
                    len(matches[m]['bf']['matches']),
                    matches[m]['bf']['outliers'],
                    matches[m]['bf']['inliers'],
                    matches[m]['bf']['time'],
                    len(matches[m]['knn_bf']['matches']),
                    matches[m]['knn_bf']['outliers'],
                    matches[m]['knn_bf']['inliers'],
                    matches[m]['knn_bf']['time'],
                    len(matches[m]['flann']['matches']),
                    matches[m]['flann']['outliers'],
                    matches[m]['flann']['inliers'],
                    matches[m]['flann']['time'],
                    ])
            
if __name__ == "__main__":

    if sys.argv < 2:
        exit(0)

    path = sys.argv[1]
    kp_algorithims = [
                ('sift', cv2.SIFT()),
                ('surf', cv2.SURF()),
                ('fast', cv2.FastFeatureDetector()),
                ('orb', cv2.ORB()),
            ]

    des_algorithims = [
                ('sift', cv2.SIFT()),
                ('surf', cv2.SURF()),
                ('orb', cv2.ORB()),
                #('brief', cv2.DescriptorExtractor_create("BRIEF")),
            ]

    benchmark = HackberryBenchmark(
            path, 
            kp_algorithims,
            des_algorithims, 
            ['jpg']
        )

    img1 = cv2.imread(os.path.join(path, 'f-0.jpg'), 0)
    img2 = cv2.imread(os.path.join(path, 'f-10.jpg'), 0)

            
    benchmark.csv_match_stats(img1, img2)
    #exit(0)

    rows = benchmark.generate_keypoints()
    rows = benchmark.generate_descriptions(rows)

    with open('stats.csv', 'w+') as f:
        csvwriter = csv.writer(f)

        csv_header = [
            'Algorithim', 
            'Image', 
            'Total Time', 
            'Time per Keypoint', 
            'Keypoints',
            'SIFT_Descriptions',
            'Description Total Time', 
            'Time per one Description', 
            'SURF_Descriptions',
            'Description Total Time', 
            'Time per one Description', 
            #'BRIEF_Descriptions',
            #'Description Total Time', 
            #'Time per one Description', 
            'ORB_Descriptions',
            'Description Total Time', 
            'Time per one Description', 
        ]

        csvwriter.writerow(csv_header)

        for row in rows:
            csvwriter.writerow([
                row['algorithim'],
                row['image'],
                row['kp_time'],
                row['time_per_keypoint'],
                row['keypoints'],
                row['sift_descriptions'],
                row['time_sift_d'],
                row['time_per_sift_d'],
                row['surf_descriptions'],
                row['time_surf_d'],
                row['time_per_surf_d'],
                #row['brief_descriptions'],
                #row['time_brief_d'],
                #row['time_per_brief_d'],
                row['orb_descriptions'],
                row['time_orb_d'],
                row['time_per_orb_d'],
            ])
