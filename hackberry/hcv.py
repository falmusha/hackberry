#!/usr/bin/python

import cv2
import threading
import time

class ImageViewer (threading.Thread):

    def __init__(self):
        super(ImageViewer, self).__init__()
        self.counter = 0

    def run(self, frame):
        s_frame = cv2.resize(frame, (0,0), fx=0.3, fy=0.3)
        cv2.imshow('out', s_frame)
        cv2.imwrite('out'+str(self.counter)+'.jpg', frame)
        self.counter += 1


class ImageBuffer (threading.Thread):

    def __init__(self, cap, buf, buf_size, lock):
        super(ImageBuffer, self).__init__()
        self._stop = threading.Event()

        self.cap = cap
        self.buf = buf
        self.buf_size = buf_size
        self.lock = lock
        self.counter = 0

    def run(self):

        while not self.stopped():
            ret, frame = self.cap.read()
            #frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if len(self.buf) < self.buf_size:
                self.lock.acquire()
                self.buf.append(frame)
                self.lock.release()
                #cv2.imwrite('f-'+str(self.counter)+'.jpg', frame)
                #self.counter += 1
                time.sleep(0.5)

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()


class ComputerVision:

    def __init__(self):
        self.x = None

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
        image = np.zeros((max(h1, h2), w1 + w2 + offset, 3), np.uint8)
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

        x_offset = 0
        y_offset = 0

        if min_x < 0:
            x_offset += -(min_x)
        if min_y < 0:
            y_offset += -(min_y)

        max_x = max(size_image1[0], max_x)
        max_y = max(size_image1[1], max_y)

        offset = (x_offset, y_offset)
        size   = (max_y, max_x)
        
        homography[0:2,2] +=  offset

        #sizes = {
                #'out': size,
                #'1': size_image1,
                #'2': img2_dims,
                #}

        return (size, offset)

    def stitch_arr(self, imgs, kp_alg, des_alg, min_match=10):

        imgs_len = len(imgs)

        if imgs_len == 0:
            return None

        out = imgs[0]
        for i in range(0, imgs_len):
            if i+1 == imgs_len:
                return out
            out = self.stitch(out, imgs[i+1], kp_alg, des_alg)

    def detect_and_compute(self, frame, kp_a, des_a):
        ''' kp_a is keypoint algorithim,
            des_a is decription algorithim
        '''
        kp = kp_a.detect(frame, None)
        return des_a.compute(frame, kp)

    def filter_matches(self, matches, ratio = 0.7):
        filtered = []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                filtered.append(m[0])
        
        return filtered

    def match(self, des1, des2):

        FLANN_INDEX_KDTREE = 0

        index_params = dict(
                algorithm = FLANN_INDEX_KDTREE,
                trees = 5
                )
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        # store all the good matches as per Lowe's ratio test.
        good = self.filter_matches(matches)

        return good

    def find_homography(self, src_pts, dst_pts):
        
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        outlier_indices = []
        for i,m in enumerate(mask):
            if m[0] == 0:
                outlier_indices.append(i)

        return (H, outlier_indices)

    def stitch(self, img1, img2, kp_alg, des_alg, min_match=1):

        kp1, des1 = self.detect_and_compute(img1, kp_alg, des_alg)
        kp2, des2 = self.detect_and_compute(img2, kp_alg, des_alg)


        matches = self.match(des1, des2)

        if len(matches)<min_match:
            print "Not enough matches are found - %d/%d" % (len(matches), min_match)
            raise Exception

        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])

        drawn_matches = self.draw_matches(img1, img2, src_pts, dst_pts)
        #self.show('matches', drawn_matches)

        #H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        (H, outlier_indices) = self.find_homography(dst_pts, src_pts)

        src_pts = np.delete(src_pts, outlier_indices, 0)
        dst_pts = np.delete(dst_pts, outlier_indices, 0)

        drawn_matches = self.draw_matches(img1, img2, src_pts, dst_pts)
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

        small_out = cv2.resize(out, (0,0), fx=0.5, fy=0.5)
        #self.show('o', small_out)

        return out

    def real_time_it(self, kp_alg, des_alg, threads, buf, min_match=1, frames=500):

        bad_frame_sequence = 4
        print "Starting in 5 seconds"
        time.sleep(5)
        print "SART!!!"
        threads['bufferer'].start()

        while len(buf) < 2:
            continue

        threads['lock'].acquire()
        out = buf.pop(0)
        threads['lock'].release()

        while frames > 0:
            if len(buf) >= 1:
                threads['lock'].acquire()
                next_frame = buf.pop(0)
                threads['lock'].release()
                old_out = out
                try:
                    out = self.stitch(out, next_frame, kp_alg, des_alg)
                    threads['viewer'].run(out)
                except Exception, e:
                    bad_frame_sequence -= 1
                    if bad_frame_sequence == 0:
                        out = buf.pop(0)
                    else:
                        out = old_out
                    print 'BAD FRAME - '+str(e)
                    continue
                print str(frames)
                frames -= 1


def test_on_files():

    kp_alg = cv2.SURF()
    des_alg = cv2.SURF()
    
    out = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/hackberry/f-0.jpg'
    out = cv2.imread(out) # queryImage

    frames_to_process = 40
    for i in range(1, frames_to_process):
        if i+1 == frames_to_process:
            break
        n_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/hackberry/f-'+str(i)+'.jpg'
        n = cv2.imread(n_path) # queryImage
        out = hcv.stitch(out, n, kp_alg, des_alg)

def rt_test():

    hcv = ComputerVision()

    kp_alg = cv2.SURF()
    des_alg = cv2.SURF()
    
    cap = cv2.VideoCapture(1)
    buf = list()
    buf_size = 6
    thread_lock = threading.Lock()

    viewer = ImageViewer()
    bufferer = ImageBuffer(cap, buf, buf_size, thread_lock)

    threads = dict()
    threads['bufferer'] = bufferer
    threads['viewer'] = viewer
    threads['lock'] = thread_lock

    try:
        hcv.real_time_it(kp_alg, des_alg, threads, buf)
        #bufferer.start()
        while True:
            continue
    except KeyboardInterrupt:
        bufferer.stop()
        bufferer.join()
        cap.release()

    bufferer.stop()
    bufferer.join()
    cap.release()

def old_test():

    img1a_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/test_images/stitching/img_1_a.jpg'
    img1b_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/test_images/stitching/img_1_b.jpg'
    img1c_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/test_images/stitching/img_1_c.jpg'
    img1d_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/test_images/stitching/img_1_d.jpg'

    img2a_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/test_images/stitching/img_2_a.jpg'
    img2b_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/test_images/stitching/img_2_b.jpg'
    img2c_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/test_images/stitching/img_2_c.jpg'

    img1a = cv2.imread(img1a_path) # queryImage
    img1b = cv2.imread(img1b_path) # trainImage
    img1c = cv2.imread(img1c_path) # trainImage
    img1d = cv2.imread(img1d_path) # trainImage


    img2a = cv2.imread(img2a_path) # queryImage
    img2b = cv2.imread(img2b_path) # trainImage
    img2c = cv2.imread(img2c_path) # trainImage

    hcv = ComputerVision()

    #kp_alg = cv2.FastFeatureDetector()
    kp_alg = cv2.SURF()
    des_alg = cv2.SURF()

    stitched_img = hcv.stitch_arr([img2a, img2b, img2c], kp_alg, des_alg)

    stitched_img = hcv.stitch_arr([img1a, img1b, img1c, img1d], kp_alg, des_alg)

def rotateImage(image, angle):

    #rotation angle in degree
    return image
    return ndimage.rotate(image, angle)

if __name__ == "__main__":

    import pdb
    import numpy as np
    import math
    from scipy import ndimage

    rt_test()

