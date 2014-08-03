#!/usr/bin/env python
import rospy
import roslib
roslib.load_manifest('flownav')

import cv2
import numpy as np
import scipy.stats as stats

from common import *
from framebuffer import FrameBuffer
from operator import attrgetter
from drone_control import DroneController
from drone_keyboard import KeyboardController


def findAppoximateScaling(queryImg, trainImg, matches, queryKPs, trainKPs, compute, showMatches=False):
    scalerange = 1 + np.arange(0.5+0.025,step=0.025)
    res = np.zeros(len(scalerange))
    expandingKPs = []

    for m in matches:
        qkp = queryKPs[m.queryIdx]
        tkp = trainKPs[m.trainIdx]

        # extract the query image patch
        x_qkp,y_qkp = qkp.pt
        r = qkp.size*1.2/9*20 // 2
        # r = qkp.size // 2
        x0,y0 = inimage(queryImg.shape,(x_qkp-r, y_qkp-r))
        x1,y1 = inimage(queryImg.shape,(x_qkp+r, y_qkp+r))
        querypatch = queryImg[y0:y1, x0:x1]

        if not querypatch.size: continue

        x_tkp,y_tkp = tkp.pt
        res[:] = np.nan     # initialize all residuals as invalid
        for i,scale in enumerate(scalerange):
            r = qkp.size*scale*1.2/9*20 // 2
            # r = qkp.size*scale // 2
            x0,y0 = inimage(trainImg.shape,(x_tkp-r, y_tkp-r))
            x1,y1 = inimage(trainImg.shape,(x_tkp+r, y_tkp+r))
            traintempl = trainImg[y0:y1, x0:x1]

            if not traintempl.size: break # feature got too large to match

            scaledtempl = cv2.resize(querypatch,traintempl.shape[::-1]
                                     , fx=scale,fy=scale
                                     , interpolation=cv2.INTER_LINEAR)

            # res[i] = cv2.matchTemplate(traintempl,scaledtempl,cv2.TM_SQDIFF)/(scale**2)
            res[i] = np.sum((scaledtempl-traintempl)**2)/(scale**2)
        if all(np.isnan(res)): continue # could not match the feature

        # determine if this is a solid match
        res_argmin = np.nanargmin(res)
        scalemin = scalerange[res_argmin]
        if scalemin > 1.2 and res[res_argmin] < 0.8*res[0]:
            r = qkp.size*scalemin*1.2/9*20 // 2
            # r = qkp.size*scalemin // 2
            ekp = copyKP(tkp)
            ekp.size = r
            expandingKPs.append(ekp)

            if not showMatches: continue

            # recalculate the best matching scaled template
            x0,y0 = roundtuple(x_tkp-r, y_tkp-r)
            x1,y1 = roundtuple(x_tkp+r, y_tkp+r)
            scaledtempl = cv2.resize(querypatch, (x1-x0,y1-y0)
                                     , fx=scale, fy=scale
                                     , interpolation=cv2.INTER_LINEAR)

            # draw the template and the best matching scaled version
            templimg = np.zeros((scaledtempl.shape[0],scaledtempl.shape[1]+querypatch.shape[1])
                                , dtype=queryImg.dtype)
            templimg[:] = 255

            drawInto(querypatch,templimg)
            drawInto(scaledtempl,templimg,tl=(querypatch.shape[1],0))
            cv2.putText(templimg,"scale=%.3f" % scalemin,(0,templimg.shape[0]-5)
                        ,cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,0,0))
            cv2.imshow('Template', templimg)

            if cv2.waitKey(0)%256 == ord('q'): showMatches=False

    return expandingKPs    


#------------------------------------------------------------------------------#
# process options and set up defaults
#------------------------------------------------------------------------------#
import optparse
import os
from subprocess import Popen
 
parser = optparse.OptionParser(usage="flownav.py [options]")
parser.add_option("-b", "--bag", dest="bag", default=None
                  , help="Use feed from a ROS bagged recording.")
parser.add_option("-m", "--show-matches", dest="showmatches"
                  , action="store_true", default=False
                  , help="Show scale matches for each feature.")
parser.add_option("--uvc", dest="uvc"
                  , action="store_true", default=False
                  , help="Use the uvc camera feed for testing purposes.")

(opts, args) = parser.parse_args()

if opts.bag:
    DEVNULL = open(os.devnull, 'wb')
    bagp = Popen(["rosbag","play",opts.bag],stdout=DEVNULL,stderr=DEVNULL)

# Initialize the drone controls and subscribe to the navdata feed
rospy.init_node("flownav", anonymous=False)
dronectrl = None if opts.uvc else DroneController() 
kbctrl =    None if opts.uvc else KeyboardController(dronectrl)
frmbuf = FrameBuffer("/uvc_camera/image_raw" if opts.uvc else "/ardrone/image_raw")

# initialize the feature description and matching methods
bfmatcher = cv2.BFMatcher()
surf_ui = cv2.SURF(1000)

# mask out a portion of the image
lastFrame = frmbuf.grab()
roi = np.zeros(lastFrame.shape,np.uint8)
scrapY, scrapX = lastFrame.shape[0]//8, lastFrame.shape[1]//16
roi[scrapY:-scrapY, scrapX:-scrapX] = True

# get keypoints and feature descriptors from query image
qkp, qdesc = surf_ui.detectAndCompute(lastFrame,roi)

#------------------------------------------------------------------------------#
# main loop
#------------------------------------------------------------------------------#
while not rospy.is_shutdown():
    currFrame = frmbuf.grab()

    if not opts.uvc:
        currNav = dronectrl.navdata

        vx, vy, vz = inttuple(*attrgetter('vx','vy','vz')(currNav))
        batt = currNav.batteryPercent
        time = currNav.header.stamp.to_sec()
        frameN = int(frmbuf.msg.header.seq)

        print time,":",(vx,vy,vz),"mm/s","(BATT:",batt,"%)"

    # get keypoints and feature descriptors from training image
    tkp, tdesc = surf_ui.detectAndCompute(currFrame,roi)

    # find the best K matches between this and last frame
    try:
        matches = bfmatcher.knnMatch(qdesc,tdesc,k=2)
    except cv2.error as e:
        print e
        continue

    # filter out poor matches by ratio test
    matches = [m[0] for m in matches if len(m)==2
               and (m[0].distance < 0.8*m[1].distance)
               and m[0].distance < 0.25]

    # assume a gaussian distribution of spatial distances between frames
    # to set a threshold for removing outliers
    # mindist = [difftuple(tkp[m.trainIdx].pt,qkp[m.queryIdx].pt) for m in matches]
    # trimdist = stats.trimboth(mindist,0.1)
    # threshdist = np.mean(trimdist) + 3*np.std(trimdist)
    threshdist = 100

    # filter out features which haven't grown
    matches = [m for m in matches
               if tkp[m.trainIdx].size > qkp[m.queryIdx].size
               and difftuple(tkp[m.trainIdx].pt,qkp[m.queryIdx].pt) < threshdist]

    # get only the keypoints that made a match
    goodKPs = tuple( (qkp[m.queryIdx],tkp[m.trainIdx]) for m in matches )
    mkp1, mkp2 = zip(*goodKPs) if goodKPs else ([],[])
    dispim = cv2.drawKeypoints(currFrame,mkp1+mkp2, None, color=(255,0,0))

    # draw the accepted matches
    map(lambda p: cv2.line(dispim,tuple(map(int,p[0].pt))
                           ,tuple(map(int,p[1].pt)),(0,255,0),2)
        , goodKPs)

    expandingKPs = findAppoximateScaling(lastFrame, currFrame, matches
                                         , qkp, tkp, surf_ui.compute,showMatches=opts.showmatches)

    # if expandingKPs: print avgtuple(map(attrgetter('pt'),expandingKPs))

    # draw the expanding keypoint matches
    cv2.drawKeypoints(dispim,expandingKPs, dispim, color=(0,0,255)
                      ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.rectangle(dispim,(scrapX,scrapY),(currFrame.shape[1]-scrapX,currFrame.shape[0]-scrapY)
                  ,(192,192,192),thickness=3)
    cv2.imshow("Match", dispim)

    k = cv2.waitKey(10)%256
    if k == ord('s'):
        fn1 = "Frame%d.png" % frameN-1
        fn2 = "Frame%d.png" % frameN
        print "saved as %s, %s..." % (fn1,fn2)
        cv2.imwrite(fn1,lastFrame)
        cv2.imwrite(fn2,currFrame)
    elif k == ord('q'):
        break
    elif not opts.uvc:
        kbctrl.keyPressEvent(k)

    lastFrame, qkp, qdesc = currFrame, tkp, tdesc

if opts.bag: bagp.kill()
cv2.destroyAllWindows()
