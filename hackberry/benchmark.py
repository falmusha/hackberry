#!/usr/bin/python

import pdb
import os
import shutil
import sys
import csv
import cv2
import numpy as np
import time
import hcv

from matplotlib import pyplot as plt

class HackberryBenchmark:

    def __init__(self, path, kp_a=[], des_a=[], image_ext=['jpg']):
        self.hcv = hcv.ComputerVision()
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

    def get_matchers(self):
        pass

    def drawMatches(self, img1, kp1, img2, kp2, matches, window_name='Matched', draw=False):
        """
        My own implementation of cv2.drawMatches as OpenCV 2.4.9
        does not have this function available but it's supported in
        OpenCV 3.0.0

        This function takes in two images with their associated 
        keypoints, as well as a list of DMatch data structure (matches) 
        that contains which keypoints matched in which images.

        An image will be produced where a montage is shown with
        the first image followed by the second image beside it.

        Keypoints are delineated with circles, while lines are connected
        between matching keypoints.

        img1,img2 - Grayscale images
        kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint 
                detection algorithms
        matches - A list of matches of corresponding keypoints through any
                OpenCV keypoint matching algorithm
        """

        # Create a new output image that concatenates the two images together
        # (a.k.a) a montage
        rows1 = img1.shape[0]
        cols1 = img1.shape[1]
        rows2 = img2.shape[0]
        cols2 = img2.shape[1]

        out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')

        # Place the first image to the left
        out[:rows1,:cols1,:] = np.dstack([img1, img1, img1])

        # Place the next image to the right of it
        out[:rows2,cols1:cols1+cols2,:] = np.dstack([img2, img2, img2])

        # For each pair of points we have between both images
        # draw circles, then connect a line between them
        for mat in matches:

            # Get the matching keypoints for each of the images
            img1_idx = mat.queryIdx
            img2_idx = mat.trainIdx

            # x - columns
            # y - rows
            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2[img2_idx].pt

            # Draw a small circle at both co-ordinates
            # radius 4
            # colour blue
            # thickness = 1
            cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
            cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)

            # Draw a line in between the two points
            # thickness = 1
            # colour blue
            cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)

        if draw:
            # Show the image
            cv2.imshow(window_name, out)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return out

    def bf_match(self, des1, des2, k, min_distance):

        if (des1.shape == (0,0))\
                or (des1.shape == (0,0)):
                    return [], 0
        bf = cv2.BFMatcher()
        t_start = time.time()
        matches = bf.knnMatch(des1, des2, k=k)
        t_end = round(time.time() - t_start, 6)

        good = []
        for m,n in matches:
            if m.distance < min_distance*n.distance:
                good.append(m)
        
        return good, t_end

    def flann_match(self, des1, des2, k, min_distance, index_params, search_params):

        if (des1.shape == (0,0))\
                or (des1.shape == (0,0)):
                    return [], 0

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        t_start = time.time()
        matches = flann.knnMatch(des1, des2, k=k)
        t_end = round(time.time() - t_start, 6)

        good = []

        matches = [match for match in matches if len(match)==2]
        for m,n in matches:
            if m.distance < min_distance*n.distance:
                good.append(m)
        
        return good, t_end

    def algorithim_name(self, a):
        return a.__class__.__name__.lower()

    def generate_matches(self, img1, img2, draw=False):

        frame1 = cv2.imread(img1, 0)
        frame2 = cv2.imread(img2, 0)

        matches = dict()
        for a in self.get_cv_kp_algorithims():

            kp1 = a[1].detect(frame1, None)
            kp2 = a[1].detect(frame2, None)

            for descriptor in self.get_cv_des_algorithims():

                d_name = descriptor[0]
                a_name = a[0]
                
                print('\n '+a_name+'/'+d_name+'\n')

                try:
                    des1 = descriptor[1].compute(frame1, kp1)[1]
                    des2 = descriptor[1].compute(frame2, kp2)[1]
                except cv2.error as e:
                    des1 = np.empty([0,0])
                    des2 = np.empty([0,0])
                    print('Something happened when computing '
                            + d_name + '/'
                            + a_name + ' discriptions')


                k = 2
                min_distance = 0.50
                bf_matches = []
                flann_matches = []

                if d_name == 'surf':
                    bf_matches, bf_t = self.bf_match(des1, des2, k, min_distance) 
                    # FLANN parameters
                    FLANN_INDEX_KDTREE = 0
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                    search_params = dict(checks=50)   # or pass empty dictionary
                    flann_matches, flann_t = self.flann_match(
                            des1, des2, k, min_distance,
                            index_params, search_params
                            )
                    #matches[a_name+'/'+d_name] = [(bf_matches, bf_t), (flann_matches, flann_t)]
                    matches[a_name+'/'+d_name] = {
                            'bf': {'matches': len(bf_matches), 'time': bf_t},
                            'flann': {'matches': len(flann_matches), 'time': flann_t}
                            }
                elif d_name == 'sift':
                    bf_matches, bf_t = self.bf_match(des1, des2, k, min_distance) 
                    # FLANN parameters
                    FLANN_INDEX_KDTREE = 0
                    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
                    search_params = dict(checks=50)   # or pass empty dictionary
                    flann_matches, flann_t = self.flann_match(
                            des1, des2, k, min_distance,
                            index_params, search_params
                            )
                    #matches[a_name+'/'+d_name] = [(bf_matches, bf_t), (flann_matches, flann_t)]
                    matches[a_name+'/'+d_name] = {
                            'bf': {'matches': len(bf_matches), 'time': bf_t},
                            'flann': {'matches': len(flann_matches), 'time': flann_t}
                            }
                elif d_name == 'orb':
                    bf_matches, bf_t = self.bf_match(des1, des2, k, min_distance) 
                    FLANN_INDEX_LSH = 6
                    search_params = dict(checks=50)   # or pass empty dictionary
                    index_params= dict(algorithm = FLANN_INDEX_LSH,
                                    table_number = 6, # 12
                                    key_size = 12,     # 20
                                    multi_probe_level = 1) #2
                    flann_matches, flann_t = self.flann_match(
                            des1, des2, k, min_distance,
                            index_params, search_params
                            )
                    matches[a_name+'/'+d_name] = {
                            'bf': {'matches': len(bf_matches), 'time': bf_t},
                            'flann': {'matches': len(flann_matches), 'time': flann_t}
                            }
                else:
                    continue

                bf_window_name = 'BF matched for '+str(len(bf_matches))+' '+a_name+'-'+d_name 
                flann_window_name = 'FLANN matched for '+str(len(bf_matches))+' '+a_name+'-'+d_name 
                bf_img = self.drawMatches(frame1, kp1, frame2, kp2, bf_matches, bf_window_name)
                flann_img = self.drawMatches(frame1, kp1, frame2, kp2, bf_matches, flann_window_name)


                result_path =  os.path.join(self.path, 'matcher', 'result')
                if not os.path.isdir(result_path):
                    os.mkdir(result_path)

                f1_n = os.path.join(result_path, bf_window_name.replace(' ', '_')+'.jpg')
                f2_n = os.path.join(result_path, flann_window_name.replace(' ', '_')+'.jpg')
                cv2.imwrite(f1_n, bf_img)
                cv2.imwrite(f2_n, flann_img)

        return matches

    def generate_descriptions(self, kps):
        for kp_row in kps:
            for d in self.get_cv_des_algorithims():
                d_name = d[0] + '_descriptions'
                des_name = d[0] + '_d'
                img = os.path.join(self.path, kp_row['image'])
                frame = cv2.imread(img, 0)
                try:
                    t_start = time.time()
                    des = d[1].compute(frame, kp_row['kps'])[1]
                    t_end = round(time.time() - t_start, 6)
                except cv2.error as e:
                    des = np.empty([0,0])
                    print('Something happened when computing '
                            + kp_row['algorithim'] + '/'
                            + d_name + ' discriptions')
                if des.shape[0]:
                    kp_row['time_'+des_name] = t_end
                    kp_row['time_per_'+des_name] = t_end/des.shape[0]
                else:
                    kp_row['time_'+des_name] = 'na'
                    kp_row['time_per_'+des_name] = 'na'
                kp_row[d_name] = str(des.shape)
                kp_row[des_name] = des

        return kps

    def generate_keypoints(self, draw=False):
        ''' Generate kps for all files in path and write them in
            f as csv file
        '''


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

                if draw:
                    self.write_img(img, frame, kp, a[0])

                rows.append(row)

        return rows
        
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
                ('brief', cv2.DescriptorExtractor_create("BRIEF")),
            ]

    benchmark = HackberryBenchmark(
            path, 
            kp_algorithims,
            des_algorithims, 
            ['jpg']
            )

    img1 = os.path.join(path, 'matcher', '1.jpg')
    img2 = os.path.join(path, 'matcher', '2.jpg')

    matches = benchmark.generate_matches(img1, img2)

    with open('matcher_stats.csv', 'w+') as f:
        csvwriter = csv.writer(f)

        csv_header = [
            'FeatureExtractor/Descriptor', 
            'BF Matches', 
            'BF Time', 
            'FLANN Matches', 
            'FLANN Time', 
        ]

        csvwriter.writerow(csv_header)
        for m in matches:
            
            csvwriter.writerow([
                m,
                matches[m]['bf']['matches'],
                matches[m]['bf']['time'],
                matches[m]['flann']['matches'],
                matches[m]['flann']['time'],
                ])
            
            
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
            'BRIEF_Descriptions',
            'Description Total Time', 
            'Time per one Description', 
            'ORB_Descriptions'
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
                row['brief_descriptions'],
                row['time_brief_d'],
                row['time_per_brief_d'],
                row['orb_descriptions'],
                row['time_orb_d'],
                row['time_per_orb_d'],
            ])
