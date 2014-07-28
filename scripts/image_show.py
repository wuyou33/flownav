#!/usr/bin/env python
import roslib
roslib.load_manifest('flownav')
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from operator import attrgetter
import time
import numpy as np

class FrameBuffer:
    '''
    FrameBuffer

    Creates a subcription node to the image publisher and converts the image
    into opencv image type.
    '''
    def __init__(self, topic="/image_raw", node="Image2cv"):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.image = None
        self.grabbed = True
        rospy.init_node(node, anonymous=False)

    def callback(self,data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data,'bgr8')
            # self.image = cv2.cvtColor(self.image,cv2.CV_8UC1)
            self.grabbed = False
        except:
            raise

    def grab(self):
        try:
            while self.grabbed is True: None
            self.grabbed = True
        except KeyboardInterrupt:
            raise
        return self.image

def drawMatch(dm,kp1,kp2,dispim):
    cv2.line(dispim
             ,tuple(map(int,kp1[dm.queryIdx].pt))
             ,tuple(map(int,kp2[dm.trainIdx].pt)),(0,255,0),2)

def cvtIdx(pt,shape):
    return int(pt[1]*shape[1] + pt[0]) if hasattr(pt, '__len__') else map(int, pt%shape[1], pt//shape[1])

def findAppoximateScaling(queryImg, trainImg, matches, queryDescs, trainDescs
                          , queryKPs, trainKPs, compute):
    trainTempImg = trainImg.copy()
    # queryTempImg = queryImg.copy()    
    scalerange = 1 + np.array(range(2,20,2))/10.
    inttuple = lambda *x: tuple(map(int,x))

    for m,querydesc,traindesc in zip(matches,queryDescs,trainDescs):
        # get training keypoint location and size
        x,y = queryKPs[m.queryIdx].pt
        r = int(queryKPs[m.queryIdx].size//2)

        # extract the query image patch
        x0,y0 = (x-r, y-r)
        x1,y1 = (x+r, y+r)
        querypatch = queryImg[y0:y1+1, x0:x1+1,:]
        cv2.rectangle(trainTempImg, inttuple(x-r,y-r), inttuple(x+r,y+r), (0,0,255), 2)
        
        # trainpatch = trainImg[y0:y1+1, x0:x1+1,:]
        # traindesc = compute(trainPatch)

        # template match patch on training image
        try:
            res = np.array(len(scalerange)+1)
            res[0] = np.sum((querydesc-traindesc)**2)
            for i,scale in enumerate(scalerange):
                scaledtempl = cv2.resize(querypatch,0,fx=scale,fy=scale)
                scaledesc = compute(scaledtempl,kp2)
                res[i+1] = np.sum((scaledesc-traindesc)**2)
                print res[i+1]

            # res = [np.sum((compute(cv2.resize(querydesc,0,fx=scale,fy=scale),queryKPs[m.queryIdx])-traindesc)**2)
            #        for scale in scalerange]

            x,y = trainKPs[m.trainIdx].pt
            scale = scalerange[np.argmin(res)]
            r *= scale
            cv2.rectangle(trainTempImg, inttuple(x-r,y-r), inttuple(x+r,y+r), (255,0,0), 2)
        except Exception, e:
            print e
            continue

        cv2.imshow('Match', trainTempImg)
        if cv2.waitKey(0)%256 == ord('q'): return

if __name__ == '__main__':
    frmbuf = FrameBuffer(topic="/uvc_camera/image_raw")

    # orb = cv2.ORB()
    bfmatcher = cv2.BFMatcher()
    surf_ui = cv2.SURF(600)

    # find the keypoints and descriptors with SIFT
    lastFrame = frmbuf.grab()
    roi = np.zeros(lastFrame.shape[:2],np.uint8)
    scrapY, scrapX = lastFrame.shape[0]//16, lastFrame.shape[1]//16

    roi[scrapY:-scrapY, scrapX:-scrapX] = True

    frameN = 1
    kp1, desc1 = surf_ui.detectAndCompute(lastFrame,roi)
    while not rospy.is_shutdown():
        frameN += 1
        currFrame = frmbuf.grab()      
        kp2, desc2 = surf_ui.detectAndCompute(currFrame,roi)

        # matches = bfmatcher.match(desc1,desc2)
        matches = bfmatcher.knnMatch(desc1,desc2,k=2)
        # matches.sort(key=attrgetter('distance'))
        # mindist = min(map(attrgetter('distance'),matches))
        # mindist = matches[0].distance
        # matches = filter(lambda x: x.distance <= 0.25, matches)
        # matches = filter(lambda x: x.distance < 2 * mindist, matches)

        # filter out poor matches by ratio test
        matches = [m[0] for m in matches if m and (m[0].distance < 0.7*m[1].distance)]
        # filter out features which haven't grown
        matches = [m for m in matches if kp2[m.trainIdx].size > kp1[m.queryIdx].size]
        if not matches: continue

        # get only the keypoints that made a match
        goodKPs = tuple( (kp1[m.queryIdx],kp2[m.trainIdx]) for m in matches )
        mkp1, mkp2 = zip(*goodKPs)
        dispim = cv2.drawKeypoints(currFrame,mkp1+mkp2, color=-1)
                                   # , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        for i,m in enumerate(matches):
            print "Match %3d: scale + %f" % (i,kp2[m.trainIdx].size-kp1[m.queryIdx].size)

        scalings = findAppoximateScaling(lastFrame, currFrame, matches, desc1, desc2, kp1, kp2, surf_ui.compute)
        # draw the matches directly from the array of Dmatches
        # map(lambda x: drawMatch(x,kp1,kp2,dispim), matches)

        # draw the accepted matches
        map(lambda p: cv2.line(dispim,tuple(map(int,p[0].pt))
                               ,tuple(map(int,p[1].pt)),(0,255,0),2)
            , goodKPs)

        cv2.imshow("Match", dispim)
        if cv2.waitKey(1)%256 == ord('s'):
            fn1 = "Frame%d.png" % frameN-1
            fn2 = "Frame%d.png" % frameN
            print "saved as %s, %s..." % (fn1,fn2)
            cv2.imwrite(fn1,lastFrame)
            cv2.imwrite(fn2,currFrame)

        lastFrame, kp1, desc1 = currFrame, kp2, desc2

    cv2.destroyAllWindows()
