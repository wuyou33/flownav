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
from drone_control import DroneController, DroneStatus
from auto_control import AutoController,CharMap,KeyMapping

from operator import itemgetter
import itertools as it
import time
import sys


def estimateKeypointExpansion(queryImg, trainImg, matches, queryKPs, trainKPs, showMatches=False,dispimg=None,winname="Match"):
    scalerange = 1 + np.arange(0.5+0.025,step=0.025)
    res = np.zeros(len(scalerange))
    expandingKPs = []

    for m in matches:
        qkp = queryKPs[m.queryIdx]
        tkp = trainKPs[m.trainIdx]

        # extract the query image patch
        x_qkp,y_qkp = qkp.pt
        r = qkp.size*1.2/9*20 // 2
        # r = qkp.size//2
        x0,y0 = trunc_coords(queryImg.shape,(x_qkp-r, y_qkp-r))
        x1,y1 = trunc_coords(queryImg.shape,(x_qkp+r, y_qkp+r))
        querypatch = queryImg[y0:y1, x0:x1]

        if not querypatch.size: continue

        x_tkp,y_tkp = tkp.pt
        res[:] = np.nan     # initialize all residuals as invalid
        for i,scale in enumerate(scalerange):
            r = qkp.size*scale*1.2/9*20 // 2
            # r = qkp.size*scale // 2
            x0,y0 = trunc_coords(trainImg.shape,(x_tkp-r, y_tkp-r))
            x1,y1 = trunc_coords(trainImg.shape,(x_tkp+r, y_tkp+r))
            traintempl = trainImg[y0:y1, x0:x1]

            if not traintempl.size: break # feature got too large to match

            scaledtempl = cv2.resize(querypatch,traintempl.shape[::-1]
                                     , fx=scale,fy=scale
                                     , interpolation=cv2.INTER_LINEAR)

            # normalize image patches before comparison
            traintempl = (traintempl - np.mean(traintempl))/np.std(traintempl)
            scaledtempl = (scaledtempl - np.mean(scaledtempl))/np.std(scaledtempl)

            # res[i] = cv2.matchTemplate(traintempl,scaledtempl,cv2.TM_CCORR_NORMED)/(scale**2)
            res[i] = np.sum(np.abs(scaledtempl-traintempl))/(scale**2)
        if all(np.isnan(res)): continue # could not match the feature

        # determine if this is a solid match
        res_argmin = np.nanargmin(res)
        scalemin = scalerange[res_argmin]
        if scalemin > 1.2 and res[res_argmin] < 0.8*res[0]:
            trainKPs[m.trainIdx].size = qkp.size*scalemin*1.2/9*20
            # trainKPs[m.trainIdx].size = qkp.size*scalemin
            expandingKPs.append(trainKPs[m.trainIdx])

            if not showMatches: continue

            # recalculate the best matching scaled template
            r = qkp.size*scalemin*1.2/9*20 // 2            
            x0,y0 = trunc_coords(trainImg.shape,(x_tkp-r, y_tkp-r))
            x1,y1 = trunc_coords(trainImg.shape,(x_tkp+r, y_tkp+r))
            traintempl = trainImg[y0:y1, x0:x1]            
            scaledtempl = cv2.resize(querypatch, traintempl.shape[::-1]
                                     , fx=scalemin, fy=scalemin
                                     , interpolation=cv2.INTER_LINEAR)

            # draw the template and the best matching scaled version
            templimg = np.zeros((scaledtempl.shape[0],scaledtempl.shape[1]+traintempl.shape[1]+querypatch.shape[1])
                                , dtype=trainImg.dtype)
            templimg[:] = 255

            drawInto(querypatch, templimg)
            drawInto(scaledtempl,templimg,tl=(querypatch.shape[1],0))
            drawInto(traintempl,templimg,tl=(querypatch.shape[1]+traintempl.shape[1],0))
            dispimg = cv2.drawKeypoints(dispimg,[trainKPs[m.trainIdx]], dispimg, color=(0,0,255)
                                        ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            cv2.imshow('Template', templimg)
            cv2.imshow(winname, dispimg)
            print "Relative scaling of template:",scalemin
            print "Find next match? ('s' to skip remaining matches,'q' to quit,enter to continue):",
            sys.stdout.flush()

            k = cv2.waitKey(1)%256
            while k not in map(ord,('\r','s','q')):
                k = cv2.waitKey(1)%256
            print
            if k == ord('s'): showMatches=False
            elif k == ord('q'): raise SystemExit

    return expandingKPs    


def generateuniqid():
    uid = 2 # starts at 2 since default class_id for keypoints can be -1 or +1
    while(1):
        yield uid
        uid += 1


#------------------------------------------------------------------------------#
# process options and set up defaults
#------------------------------------------------------------------------------#
import optparse
import os
from subprocess import Popen
 
parser = optparse.OptionParser(usage="flownav.py [options]")
parser.add_option("-b", "--bag", dest="bag", default=None
                  , help="Use feed from a ROS bagged recording.")
parser.add_option("--threshold", dest="threshold", type=float,default=4000.
                  , help="Use feed from a ROS bagged recording.")
parser.add_option("-m", "--draw-scale-match", dest="showmatches"
                  , action="store_true", default=False
                  , help="Show scale matches for each expanding keypoint.")
parser.add_option("--no-draw", dest="nodraw"
                  , action="store_true", default=False
                  , help="Draw keypoints and their matches.")
parser.add_option("--cam-topic", dest="camtopic", default="/ardrone"
                  , help="Use another camera topic for testing purposes.")

(opts, args) = parser.parse_args()

if opts.bag:
    DEVNULL = open(os.devnull, 'wb')
    bagp = Popen(["rosbag","play",opts.bag],stdout=DEVNULL,stderr=DEVNULL)

# Initialize the drone controls and subscribe to the navdata feed
rospy.init_node("flownav", anonymous=False)
dronectrl = None if opts.camtopic != "/ardrone" else DroneController() 
autoctrl = None if opts.camtopic != "/ardrone" else AutoController(dronectrl)
frmbuf = FrameBuffer(opts.camtopic+"/image_raw")

# initialize the feature description and matching methods
bfmatcher = cv2.BFMatcher()
surf_ui = cv2.SURF(hessianThreshold=opts.threshold)

# mask out a portion of the image
lastFrame = frmbuf.grab()
roi = np.zeros(lastFrame.shape,np.uint8)
scrapY, scrapX = lastFrame.shape[0]//8, lastFrame.shape[1]//16
roi[scrapY:-scrapY, scrapX:-scrapX] = True

# get the keypoints corresponding to a match
getMatchKPs = lambda x: (queryKP[x.queryIdx],trainKP[x.trainIdx])

# get keypoints and feature descriptors from query image
queryKP, qdesc = surf_ui.detectAndCompute(lastFrame,roi)

multimatches = {}

idgen = generateuniqid()
getuniqid = lambda : idgen.next()

# Print intro output to user
print "Options"
print "-"*len("Options")
print "- Subscribed to", repr(opts.camtopic)
print "- Hessian threshold set at", repr(opts.threshold)
print

print "-"*40
cv2.namedWindow("Match", flags=cv2.WINDOW_OPENGL|cv2.WINDOW_NORMAL)
if opts.showmatches: cv2.namedWindow("Template", flags=cv2.WINDOW_OPENGL|cv2.WINDOW_NORMAL)
print "-"*40
print

print "Keyboard Controls for automated controller"
print "-"*len("Keyboard Controls for automated controller")
for k,v in CharMap.items():
    print k.ljust(20),'=',repr(v).ljust(5)
print

print
print "* Press 'q' at any time to quit"
print "* Press 'd' at any time to toggle keypoint drawing"
print "* Press 'm' at any time to toggle scale matching drawing"
print

#------------------------------------------------------------------------------#
# main loop
#------------------------------------------------------------------------------#
frameN = 1
while not rospy.is_shutdown():
    currFrame = frmbuf.grab()
    t1 = time.time()
    frameN = frmbuf.msg.header.seq
    frametime =  frmbuf.msg.header.stamp.to_sec()

    ### Find keypoint matches for this frame and filter them
    # get keypoints and feature descriptors from training image
    trainKP, tdesc = surf_ui.detectAndCompute(currFrame,roi)

    # find the best K matches between this and last frame
    try:
        matches = bfmatcher.knnMatch(qdesc,tdesc,k=2)
    except cv2.error as e:
        print e
        matches = []

    # assume a gaussian distribution of spatial distances between frames
    # to set a threshold for removing outliers
    # mindist = [diffKP_L2(*getMatchKPs(m[0])) for m in matches]
    # trimdist = stats.trimboth(mindist,0.1)
    # threshdist = np.mean(trimdist) + np.std(trimdist)
    threshdist = 100

    # filter out poor matches by ratio test, maximum (descriptor) distance, and
    # maximum spatial distance
    matches = [m[0] for m in matches
               if len(m)==2
               and m[0].distance < 0.8*m[1].distance
               and m[0].distance < 0.25
               and diffKP_L2(*getMatchKPs(m[0])) < threshdist]


    ### Update the keypoint history
    # add these keypoints to the feature history list
    for m in matches:
        if queryKP[m.queryIdx].class_id in multimatches:
            trainKP[m.trainIdx].class_id = queryKP[m.queryIdx].class_id
        else:
            trainKP[m.trainIdx].class_id = getuniqid()
            multimatches[trainKP[m.trainIdx].class_id] = KeyPoint(trainKP[m.trainIdx])            

        multimatches[trainKP[m.trainIdx].class_id].detects += 1
        multimatches[trainKP[m.trainIdx].class_id].age = -1
    # discard keypoints that haven't been found recently
    for clsid in multimatches.keys():
        multimatches[clsid].age += 1
        if multimatches[clsid].age > MAX_AGE: del multimatches[clsid]

    # get only the keypoints that made a match and are not expanding
    matchKPs = tuple(getMatchKPs(m) for m in matches)
                     # if multimatches[trainKP[m.trainIdx].class_id].detects > 3)

    ### Find expanding keypoints
    # filter out features which haven't grown
    matches = [m for m in matches
               if trainKP[m.trainIdx].size > queryKP[m.queryIdx].size]
               # and multimatches[trainKP[m.trainIdx].class_id].detects > 3]

    ### Draw matches
    dispim = dispim if not opts.nodraw else currFrame
    if not opts.nodraw:
        # draw each key point and a line between their match
        mkp1, mkp2 = zip(*matchKPs) if matchKPs else ([],[])
        dispim = cv2.drawKeypoints(currFrame,mkp1+mkp2, None, color=(255,0,0))
        map(lambda p: cv2.line(dispim, inttuple(*p[0].pt), inttuple(*p[1].pt), (0,255,0), 2), matchKPs)

        # draw the search bounding box
        cv2.rectangle(dispim,(scrapX,scrapY),(currFrame.shape[1]-scrapX,currFrame.shape[0]-scrapY)
                      ,(192,192,192),thickness=2)

    expandingKPs = estimateKeypointExpansion(lastFrame, currFrame, matches
                                             , queryKP, trainKP
                                             , showMatches=opts.showmatches
                                             , dispimg=dispim
                                             , winname="Match")
    print time.time() - t1

    # draw each expanding key point
    if not opts.showmatches:
        cv2.drawKeypoints(dispim,expandingKPs, dispim, color=(0,0,255)
                          ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    
    if dronectrl:
        vx, vy, vz = inttuple(*attrgetter('vx','vy','vz')(dronectrl.navdata))
        batt = dronectrl.navdata.batteryPercent
        stat = "BATT=%.2f" % (batt)
        cv2.putText(dispim,stat,(10,currFrame.shape[0]-20)
                    ,cv2.FONT_HERSHEY_TRIPLEX, 0.65, (0,0,255))
    cv2.imshow("Match", dispim)

    indanger = False
    x,y = avgKP(expandingKPs) if expandingKPs else 0,0
    if indanger:
        if x < currFrame.shape[1]:  dronectrl.EvadeRight()
        else:                       dronectrl.EvadeLeft()

    k = cv2.waitKey(1)%256
    if autoctrl and k in KeyMapping:    autoctrl.keyPressEvent(k)
    elif k == ord('m'):                 opts.showmatches ^= True
    elif k == ord('d'):                 opts.nodraw ^= True
    elif k == ord('q'):                 break

    lastFrame, queryKP, qdesc = currFrame, trainKP, tdesc

if opts.bag: bagp.kill()
cv2.destroyAllWindows()
