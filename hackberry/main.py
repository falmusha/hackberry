#!/usr/local/bin/python

import hackberry_serial as hackberry_s
import hcv as hackberry_cv
import cv2
import threading
import time

class StitchBuffer (threading.Thread):

    def __init__(self, kp_alg, des_alg, buf, buf_size, lock):
        super(StitchBuffer, self).__init__()
        self._stop = threading.Event()
        self.hcv = hackberry_cv.ComputerVision()
        self.kp_alg = kp_alg
        self.des_alg = des_alg
        self.buf = buf
        self.buf_size = buf_size
        self.lock = lock
        self.mosaic_counter = 0
        self.counter = 0
        self.start_mosaic = False
        self.viewer = ImageViewer()


    def show_img(self, name, img):
        cv2.imwrite(name+'.jpg', img)

    def create_mosaic(self, img):

        try:
            out = self.hcv.stitch(self.mosaic, img, self.kp_alg, self.des_alg)
            print 'Created Mosaic'
            self.mosaic_counter += 1
            self.show_img('mosaic-'+str(self.mosaic_counter), out)
        except Exception, e:
            print 'STITCH_BUFFER_MOSAIC: BAD FRAME - '+str(e)

    def stitch(self, img, next_img):

        try:
            out = self.hcv.stitch(img, next_img, self.kp_alg, self.des_alg)
            self.counter += 1
            self.show_img('stitch-'+str(self.mosaic_counter), out)
        except Exception, e:
            print 'STITCH_BUFFER: BAD FRAME - '+str(e)
            out = next_img

        return out

    def empty_buffer(self):

        self.lock.acquire()
        img1 = self.buf.pop(0)
        img2 = self.buf.pop(0)
        self.lock.release()
        out = self.stitch(img1, img2)

        for i in range(self.buf_size-2):
            self.lock.acquire()
            next_img = self.buf.pop(0)
            self.lock.release()
            out = self.stitch(out, next_img)

        if self.start_mosaic:
            self.create_mosaic(out)
        else:
            self.mosaic = out
            self.start_mosaic = True

    def run(self):

        while not self.stopped():
            if len(self.buf) >= self.buf_size:
                print 'STITCH_BUFFER: EMPTYING BUFFER'
                self.empty_buffer()

    
    def stop(self):
        print 'StitchBuffer is stopped'
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

class ImageBuffer (threading.Thread):

    def __init__(self, cap, buf, buf_size, lock):
        super(ImageBuffer, self).__init__()
        self._stop = threading.Event()

        self.cap = cap
        self.buf = buf
        self.buf_size = buf_size
        self.lock = lock
        self.counter = 0

    def show_img(self, img):
        s_img = cv2.resize(img, (0,0), fx=0.2, fy=0.2)

    def run(self):

        while not self.stopped():
            ret, frame = self.cap.read()
            frame = cv2.resize(frame, (0,0), fx=0.3, fy=0.3)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #cv2.imshow('cap', frame)
            #self.show_img(gray)

            if len(self.buf) < self.buf_size:
                self.lock.acquire()
                self.buf.append(frame)
                self.lock.release()
                #cv2.imwrite('f-'+str(self.counter)+'.jpg', frame)
                #self.counter += 1
                time.sleep(0.2)

    def stop(self):
        print 'ImageBuffer is stopped'
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

class ImageViewer (threading.Thread):

    def __init__(self):
        super(ImageViewer, self).__init__()
        self._stop = threading.Event()

    def show_img(self, img):
        cv2.imshow('_out', img)

    def run(self, img):
        self.show_img(img)

    def show_kp(self, img, alg):
        kp = alg.detect(img, None)
        img_kp = cv2.drawKeypoints(img, kp, color=(255,0,0))
        cv2.destroyWindow('_out_kp')
        cv2.imshow('_out_kp', img_kp)


def rt_blocking_pop(buf, lock):

    while len(buf) < 1:
        continue

    lock.acquire()
    item = buf.pop(0)
    lock.release()

    return item

def rt_stitch(hcv, img, kp_alg, des_alg, cam_buf, cam_lock, stitch_buf, stitch_lock):


    next_img = rt_blocking_pop(cam_buf, cam_lock)

    done = False
    try:
        out = hcv.stitch(img, next_img, kp_alg, des_alg)
        done = True
        #stitch_lock.acquire()
        #stitch_buf.append(out)
        #stitch_lock.release()
    except Exception, e:
        print 'BAD FRAME - '+str(e)
        out = img

    return (out, done) 

def rt_it_slow(hcv, kp_alg, des_alg, threads, cam_buf, stitch_buf, frames=200):

    print "Starting in 5 seconds"
    time.sleep(5)
    print "START!!!"

    stitched = 0
    threads['bufferer'].start()
    #threads['stitcher'].start()

    out = rt_blocking_pop(cam_buf, threads['cam_lock'])

    threads['viewer2'].show_kp(out, kp_alg)

    while True:
        tries = 10
        while tries > 0:
            (out, done) = rt_stitch(
                        hcv,
                        out, 
                        kp_alg, 
                        des_alg, 
                        cam_buf, 
                        threads['cam_lock'],
                        stitch_buf,
                        threads['stitch_lock']
                    )
            if done:
                print '.',
                tries = 0
                threads['viewer'].run(out)
                stitched += 1
            else:
                tries -= 1
                if tries == 0:
                    print 'Start New sequence'
                    out = rt_blocking_pop(cam_buf, threads['cam_lock'])
                    threads['viewer2'].show_kp(out, kp_alg)

    print 'Done!!'

def rt_it(hcv, kp_alg, des_alg, threads, cam_buf,  stitch_buf, frames=50):

    print "Starting in 5 seconds"
    time.sleep(5)
    print "START!!!"

    stitched = 0
    threads['bufferer'].start()
    threads['stitcher'].start()

    out = rt_blocking_pop(cam_buf, threads['cam_lock'])

    while stitched < frames:
        tries = 4
        while tries > 0:
            #print 'Try '+str(tries)
            (out, done) = rt_stitch(
                        hcv,
                        out, 
                        kp_alg, 
                        des_alg, 
                        cam_buf, 
                        threads['cam_lock'],
                        stitch_buf,
                        threads['stitch_lock']
                    )
            if done:
                tries = 0
                out = rt_blocking_pop(cam_buf, threads['cam_lock'])
            else:
                tries -= 1

        stitched += 1
        threads['viewer'].run(out)
        #print 'Stitched ' + str(stitched) + ' frames'

def show(name, img):
    cv2.imshow(name, img)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows() 

def rt_test():

    hcv = hackberry_cv.ComputerVision()

    kp_alg = cv2.SURF()
    #kp_alg = cv2.FastFeatureDetector()
    des_alg = cv2.SURF()
    
    cap = cv2.VideoCapture(0)
    cam_buf = list()
    cam_buf_size = 10
    cam_thread_lock = threading.Lock()

    stitch_buf = list()
    stitch_buf_size = 10
    stitch_thread_lock = threading.Lock()

    bufferer = ImageBuffer(cap, cam_buf, cam_buf_size, cam_thread_lock)
    stitcher = StitchBuffer(kp_alg, des_alg, stitch_buf, stitch_buf_size, stitch_thread_lock)
    viewer = ImageViewer()
    viewer2 = ImageViewer()

    threads = dict()
    threads['bufferer'] = bufferer
    threads['stitcher'] = stitcher
    threads['viewer'] = viewer
    threads['viewer2'] = viewer2
    threads['cam_lock'] = cam_thread_lock
    threads['stitch_lock'] = stitch_thread_lock

    try:
        #rt_it(hcv, kp_alg, des_alg, threads, cam_buf,  stitch_buf)
        rt_it_slow(hcv, kp_alg, des_alg, threads, cam_buf,  stitch_buf)
        while True:
            continue
    except KeyboardInterrupt:
        #stitcher.stop()
        #stitcher.join()
        bufferer.stop()
        bufferer.join()
        cap.release()

    #stitcher.stop()
    #stitcher.join()
    bufferer.stop()
    bufferer.join()
    cap.release()

def uno_rt_it(hcv, img, kp_alg, des_alg, cam_buf, frames=50):

    while len(cam_buf) < 1:
        continue

    next_img = cam_buf.pop(0)

    try:
        return hcv.stitch(img, next_img, kp_alg, des_alg)
    except Exception:
        return next_img

def rt_test_uno():

    print "Starting in 5 seconds"
    time.sleep(5)
    print "START!!!"

    hcv = hackberry_cv.ComputerVision()

    kp_alg = cv2.SURF()
    des_alg = cv2.SURF()
    
    cam_buf = list()
    bufferer = hackberry_s.UnoBuffer(cam_buf)
    bufferer.start()

    while len(cam_buf) < 1:
        continue

    out = cam_buf.pop(0)

    try:
        while True:
            out = uno_rt_it(hcv, out, kp_alg, des_alg, cam_buf)
            cv2.imshow('uno_out', out)
    except KeyboardInterrupt:
        bufferer.stop()
        bufferer.join()
        return

    bufferer.stop()
    bufferer.join()

def test_on_files():

    hcv = hackberry_cv.ComputerVision()

    kp_alg = cv2.SURF()
    des_alg = cv2.SURF()
    
    #out = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/hackberry/stitch-0.jpg'
    out = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/hackberry/f64/f-0.jpg'
    out = cv2.imread(out) # queryImage

    frames_to_process = 40
    for i in range(1, frames_to_process):
        print '-------------------------------------------'
        if i+1 == frames_to_process:
            break
        if i == 31:
            break
        n_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/hackberry/f64/f-'+str(i)+'.jpg'
        print n_path
        print '-------------------------------------------'
        #n_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/hackberry/stitch-'+str(i)+'.jpg'
        n = cv2.imread(n_path) # queryImage
        t_s = time.time()
        out = hcv.stitch(out, n, kp_alg, des_alg)
        t_e = time.time() - t_s
        print 'T_TOTA = '+str(t_e)
        #small_out = cv2.resize(out, (0,0), fx=0.3, fy=0.3)
        #show('out', small_out)

    cv2.imwrite('out.jpg', out)

def old_test():

    img1a_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/test_images/images/img_1_a.jpg'
    img1b_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/test_images/images/img_1_b.jpg'
    img1c_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/test_images/images/img_1_c.jpg'
    img1d_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/test_images/images/img_1_d.jpg'

    img2a_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/test_images/images/img_2_a.jpg'
    img2b_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/test_images/images/img_2_b.jpg'
    img2c_path = '/Users/ifahad7/Dropbox/School/FYDP/hackberry/test_images/images/img_2_c.jpg'

    img1a = cv2.imread(img1a_path) # queryImage
    img1b = cv2.imread(img1b_path) # trainImage
    img1c = cv2.imread(img1c_path) # trainImage
    img1d = cv2.imread(img1d_path) # trainImage


    img2a = cv2.imread(img2a_path) # queryImage
    img2b = cv2.imread(img2b_path) # trainImage
    img2c = cv2.imread(img2c_path) # trainImage

    hcv = hackberry_cv.ComputerVision()

    kp_alg = cv2.FastFeatureDetector()
    #kp_alg = cv2.SURF()
    des_alg = cv2.SURF()


    out = hcv.stitch(img2a, img2b, kp_alg, des_alg)
    out = hcv.stitch(out, img2c, kp_alg, des_alg)
    cv2.imwrite('out_1.jpg', out)

    out = hcv.stitch(img1a, img1b, kp_alg, des_alg)
    out = hcv.stitch(out, img1c, kp_alg, des_alg)
    out = hcv.stitch(out, img1d, kp_alg, des_alg)
    cv2.imwrite('out_2.jpg', out)

if __name__ == "__main__":

    #old_test()
    #test_on_files()
    #rt_test()
    rt_test_uno()
