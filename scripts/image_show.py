#!/usr/bin/env python
import copy
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

def findAppoximateScaling(queryImg, trainImg, matches, queryKPs, trainKPs, compute):
    trainTempImg = trainImg.copy()
    queryTempImg = queryImg.copy()    
    scalerange = 1+np.array(range(10))/10.
    inttuple = lambda *x: tuple(map(int,x))
    roundtuple = lambda *x: tuple(map(lambda y: int(round(y)),x))
    res = np.zeros(len(scalerange))
    
    for m in matches:
        qkp = queryKPs[m.queryIdx]
        tkp = trainKPs[m.trainIdx]        

        # extract the query image patch
        x,y = qkp.pt
        r = qkp.size*1.2*9/10

        x0,y0 = roundtuple(x-r, y-r)
        x1,y1 = roundtuple(x+r, y+r)
        querypatch = queryImg[y0:y1+1, x0:x1+1,:]
        cv2.rectangle(queryTempImg, (x0,y0), (x1,y1), (255,0,0), 2)

        ckp = copy.copy(qkp)
        ckp.pt = (qkp.pt[0]-x0, qkp.pt[1]-y0)
        dispim = cv2.drawKeypoints(querypatch,[ckp], None, color=-1)
        cv2.imshow('Template', dispim)
        print "Template shape:", querypatch.shape

        x_tkp,y_tkp = tkp.pt
        for i,scale in enumerate(scalerange):
            r = qkp.size*scale*1.2*9/10
            x0,y0 = roundtuple(x_tkp-r, y_tkp-r)
            x1,y1 = roundtuple(x_tkp+r, y_tkp+r)
            traintempl = trainImg[y0:y1+1, x0:x1+1,:]
            dshape = traintempl.shape[1::-1] # reverse of the first 2 coords
            scaledtempl = cv2.resize(querypatch,dshape
                                     ,fx=scale,fy=scale, interpolation=cv2.INTER_LINEAR)
            # skp = copy.copy(tkp)
            # skp.pt = roundtuple(tkp.pt[0]-x0, tkp.pt[1]-y0)
            # scaledesc = compute(scaledtempl,[skp])[1]
            # traindesc = compute(traintempl,[skp])[1]
            # res[i] = np.sum((scaledesc-traindesc)**2)/(scale**2)
            res[i] = np.sum(scaledtempl-traintempl)/(scale**2)

            cv2.imshow('Scaled', scaledtempl)
            if cv2.waitKey(0)%256 == ord('q'): break
                            
        # res = [np.sum((compute(cv2.resize(querydesc,0,fx=scale,fy=scale),qkp)-traindesc)**2)
        #        for scale in scalerange]

        x,y = trainKPs[m.trainIdx].pt
        scale = scalerange[np.argmin(res)]
        r = qkp.size*scale*1.2*9/10
        print "Match scale:", scale
        cv2.rectangle(trainTempImg, roundtuple(x-r,y-r), roundtuple(x+r,y+r), (0,0,255), 2)

        cv2.imshow('Match', trainTempImg)
        if cv2.waitKey(0)%256 == ord('q'): return


if __name__ == '__main__':
    # subscribe to the camera feed and grab the first frame
    frmbuf = FrameBuffer(topic="/uvc_camera/image_raw")
    lastFrame = frmbuf.grab()

    # initialize the feature description and matching methods
    bfmatcher = cv2.BFMatcher()
    surf_ui = cv2.SURF(600)

    # mask out a portion of the image
    roi = np.zeros(lastFrame.shape[:2],np.uint8)
    scrapY, scrapX = lastFrame.shape[0]//8, lastFrame.shape[1]//8
    roi[scrapY:-scrapY, scrapX:-scrapX] = True

    # get keypoints and feature descriptors from query image
    qkp, qdesc = surf_ui.detectAndCompute(lastFrame,roi)
    
    frameN = 1
    while not rospy.is_shutdown():
        frameN += 1
        currFrame = frmbuf.grab()

        # get keypoints and feature descriptors from training image
        tkp, tdesc = surf_ui.detectAndCompute(currFrame,roi)

        # find the best K matches between this and last frame
        # matches = bfmatcher.match(qdesc,tdesc)
        matches = bfmatcher.knnMatch(qdesc,tdesc,k=2)
        # matches.sort(key=attrgetter('distance'))
        # mindist = min(map(attrgetter('distance'),matches))
        # mindist = matches[0].distance
        # matches = filter(lambda x: x.distance <= 0.25, matches)
        # matches = filter(lambda x: x.distance < 2 * mindist, matches)

        # filter out poor matches by ratio test
        matches = [m[0] for m in matches if m and (m[0].distance < 0.7*m[1].distance)]
        # filter out features which haven't grown
        matches = [m for m in matches if tkp[m.trainIdx].size > qkp[m.queryIdx].size]
        if not matches: continue

        # get only the keypoints that made a match
        goodKPs = tuple( (qkp[m.queryIdx],tkp[m.trainIdx]) for m in matches )
        mkp1, mkp2 = zip(*goodKPs)
        dispim = cv2.drawKeypoints(currFrame,mkp1+mkp2, None, color=-1)
                                   # , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # for i,m in enumerate(matches):
        #     print "Match %3d: scale + %f" % (i,tkp[m.trainIdx].size-qkp[m.queryIdx].size)

        scalings = findAppoximateScaling(lastFrame, currFrame, matches, qkp, tkp, surf_ui.compute)
        # draw the matches directly from the array of Dmatches
        # map(lambda x: drawMatch(x,qkp,tkp,dispim), matches)

        # draw the accepted matches
        map(lambda p: cv2.line(dispim,tuple(map(int,p[0].pt))
                               ,tuple(map(int,p[1].pt)),(0,255,0),2)
            , goodKPs)

        cv2.imshow("Match", dispim)
        k = cv2.waitKey(1)%256
        if k == ord('s'):
            fn1 = "Frame%d.png" % frameN-1
            fn2 = "Frame%d.png" % frameN
            print "saved as %s, %s..." % (fn1,fn2)
            cv2.imwrite(fn1,lastFrame)
            cv2.imwrite(fn2,currFrame)
        elif k == ord('q'):
            break

        lastFrame, qkp, qdesc = currFrame, tkp, tdesc

    cv2.destroyAllWindows()
