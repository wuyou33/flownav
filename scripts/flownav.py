#!/usr/bin/env python
import roslib
roslib.load_manifest('flownav')
import sys
import rospy
import cv2
from operator import attrgetter
import numpy as np
from subscribers import FrameBuffer
from common import *


def findAppoximateScaling(queryImg, trainImg, matches, queryKPs, trainKPs, compute):
    scalerange = 1+np.array(range(10))/10.
    res = np.zeros(len(scalerange))
    expandingKPs = []

    for m in matches:
        qkp = queryKPs[m.queryIdx]
        tkp = trainKPs[m.trainIdx]

        # extract the query image patch
        x_qkp,y_qkp = qkp.pt
        r = qkp.size*1.2/9*20 // 2
        x0,y0 = roundtuple(x_qkp-r, y_qkp-r)
        x1,y1 = roundtuple(x_qkp+r, y_qkp+r)
        querypatch = queryImg[y0:y1, x0:x1]

        x_tkp,y_tkp = tkp.pt
        for i,scale in enumerate(scalerange):
            r = qkp.size*scale*1.2/9*20 // 2
            x0,y0 = roundtuple(x_tkp-r, y_tkp-r)
            x1,y1 = roundtuple(x_tkp+r, y_tkp+r)
            traintempl = trainImg[y0:y1, x0:x1]

            # print "Expected scale:", querypatch.shape[0]*scale, querypatch.shape[1]*scale
            # print "Forced scale:", traintempl.shape
            try:
                scaledtempl = cv2.resize(querypatch,traintempl.shape[::-1]
                                         , fx=scale,fy=scale
                                         , interpolation=cv2.INTER_LINEAR)
                res[i] = np.sum(scaledtempl-traintempl)/(scale**2)
            except:
                break
            # print
        else:
            # determine if this is a solid match
            sorted_res = np.sort(res)
            if sorted_res[0] > 1.2 and sorted_res[0] < 0.8*sorted_res[1]:
                scale = scalerange[np.argmin(res)]
                tkp.size = qkp.size*scale*1.2/9*20 // 2
                expandingKPs.append(tkp)

                # # recalculate the best matching scaled template
                # r = qkp.size*scale*1.2*9/10
                # x0,y0 = roundtuple(x_tkp-r, y_tkp-r)
                # x1,y1 = roundtuple(x_tkp+r, y_tkp+r)
                # traintempl = trainImg[y0:y1+1, x0:x1+1]
                # scaledtempl = cv2.resize(querypatch, traintempl.shape[::-1]
                #                          , fx=scale, fy=scale
                #                          , interpolation=cv2.INTER_LINEAR)

                # # # draw the template and the best matching scaled version
                # templimg = np.zeros((scaledtempl.shape[0],scaledtempl.shape[1]+querypatch.shape[1])
                #                     , dtype=queryImg.dtype)
                # templimg[:] = 255

                # drawInto(querypatch,templimg)
                # drawInto(scaledtempl,templimg,tl=(querypatch.shape[1],0))
                # cv2.imshow('Template', templimg)
                # print "Template shape:", querypatch.shape
                # print "Match scale:", scale
                
                #if cv2.waitKey(0)%256 == ord('q'): return

    return expandingKPs    


# subscribe to the camera feed and grab the first frame
frmbuf = FrameBuffer(topic="/uvc_camera/image_raw")
lastFrame = frmbuf.grab()

# initialize the feature description and matching methods
bfmatcher = cv2.BFMatcher()
surf_ui = cv2.SURF(400)

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
    dispim = cv2.drawKeypoints(currFrame,mkp1+mkp2, None, color=(255,0,0))
                               # , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # for i,m in enumerate(matches):
    #     print "Match %3d: scale + %f" % (i,tkp[m.trainIdx].size-qkp[m.queryIdx].size)

    # draw the accepted matches
    map(lambda p: cv2.line(dispim,tuple(map(int,p[0].pt))
                           ,tuple(map(int,p[1].pt)),(0,255,0),2)
        , goodKPs)

    expandingKPs = findAppoximateScaling(lastFrame, currFrame, matches
                                     , qkp, tkp, surf_ui.compute)

    # draw the expanding keypoint matches
    # map(lambda x: drawMatch(x,qkp,tkp,dispim,color=(0,0,255)), expandingKPs)
    cv2.drawKeypoints(dispim,expandingKPs, dispim, color=(0,0,255)
                      ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("Match", dispim)

    k = cv2.waitKey(0 if expandingKPs else 100)%256
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
