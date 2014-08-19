#!/usr/bin/env python
import rospy
from std_srvs.srv import Empty

import cv2
import numpy as np
import scipy.stats as stats

from common import *
from framebuffer import ROSCamBuffer,VideoBuffer
from operator import attrgetter,itemgetter
from keyboard_control import KeyboardController,CharMap,KeyMapping
from collections import OrderedDict

import time,sys


TEMPLATE_WIN = "Template matching"
MIN_THRESH = 500
MAX_THRESH = 2500
TARGET_N_KP = 100

gmain_win = "flownav"


def React(controller,img,timediff):
    v = controller.navdata.vx

    ttc = timediff/relsize
    if ttc <= threshttc:
        if (x_obs-img.shape[1]//2) < 0: self.RollRight()
        if x_obs >= img.shape[1]//2: self.RollLeft()


def ClusterKeypoints(keypoints,img):
    if len(keypoints) < 2: return
    # cluster overlapping features
    cluster = []
    unclusteredKPs = sorted(keypoints,key=attrgetter('pt'))
    
    while unclusteredKPs:
        clust = [unclusteredKPs.pop(0)]
        kp = clust[0]
        i = 0
        # print "Cluster %d, i=%d" % (len(cluster),i)
        # print kp.pt, kp.size//2
        while i < len(unclusteredKPs):
            if overlap(kp,unclusteredKPs[i]):
                # print "Cluster %d, i=%d, d=%d" % (len(cluster),i,diffKP_L2(kp,unclusteredKPs[i]))
                # print unclusteredKPs[i].pt, unclusteredKPs[i].size//2
                clust.append(unclusteredKPs.pop(i))
            else:
                i += 1
        if len(clust)>=2: cluster.append(Cluster(clust,img))
        # else: print "Cluster %d discarded" % len(cluster)

    newcluster = []
    while cluster:
        cl = cluster.pop(0)
        newclust = cl.KPs
        i = 0
        while i < len(cluster):
            if bboverlap(cl,cluster[i]):
                newcl = cluster.pop(i)
                newclust += newcl.KPs
            else:
                i += 1
        newcluster.append(Cluster(newclust,img))

    # print
    return newcluster

# def Cluster(keypoints,img):
#     unclusteredKPs = sorted(keypoints,key=attrgetter('pt'))

#     cluster = [ClusterRecurs(kp[i],keypoints[i+1:]) for i in range(len(unclusteredKPs))]
        
    

# def ClusterRecurs(refkp,keypoints):
#     if len(keypoints) == 0: return []

#     newkp = keypoints[0] if overlap(refkp,keypoints[0]) else refkp
    
#     return ClusterRecurs(newkp,keypoints[1:])


def estimateKeypointExpansion(queryImg, trainImg, matches, queryKPs, trainKPs
                              , showMatches=False, dispimg=None, method='L2'):
    expandingKPs = []
    scale_argmin = []
    ismatch = np.zeros(len(matches),np.bool)

    scalerange = 1.0+np.arange(0.75+0.0125,step=0.0125)
    res = np.zeros(len(scalerange))
    k = None
    for idx,m in enumerate(matches):
        qkp = queryKPs[m.queryIdx]
        tkp = trainKPs[m.trainIdx]

        # /* Extract the query and train image patch and normalize them. */ #
        x_qkp,y_qkp = qkp.pt
        r = qkp.size*1.2/9*20 // 2
        # r = qkp.size // 2
        x0,y0 = trunc_coords(queryImg.shape,(x_qkp-r, y_qkp-r))
        x1,y1 = trunc_coords(queryImg.shape,(x_qkp+r, y_qkp+r))
        querypatch = queryImg[y0:y1, x0:x1]
        if not querypatch.size: continue
        querypatch = (querypatch-np.mean(querypatch))/np.std(querypatch)

        x_tkp,y_tkp = tkp.pt
        r = qkp.size*scalerange[-1]*1.2/9*20 // 2        
        # r = qkp.size*scalerange[-1] // 2
        x0,y0 = trunc_coords(trainImg.shape,(x_tkp-r, y_tkp-r))
        x1,y1 = trunc_coords(trainImg.shape,(x_tkp+r, y_tkp+r))
        trainpatch = trainImg[y0:y1, x0:x1]
        if not trainpatch.size: continue
        trainpatch = (trainpatch-np.mean(trainpatch))/np.std(trainpatch)

        # /* Scale up the query to perform template matching */ #
        x_tkp,y_tkp = x_tkp-x0,y_tkp-y0
        res[:] = np.nan
        for i,scale in enumerate(scalerange):
            r = qkp.size*scale*1.2/9*20 // 2
            # r = qkp.size*scale // 2
            x0,y0 = trunc_coords(trainpatch.shape,(x_tkp-r, y_tkp-r))
            x1,y1 = trunc_coords(trainpatch.shape,(x_tkp+r, y_tkp+r))
            scaledtrain = trainpatch[y0:y1, x0:x1]
            if not scaledtrain.size: continue

            scaledquery = cv2.resize(querypatch,scaledtrain.shape[::-1]
                                     , fx=scale,fy=scale
                                     , interpolation=cv2.INTER_LINEAR)

            if 'corr' in method:
                res[i] = np.sum(scaledquery*scaledtrain)
            elif 'L1' in method:
                res[i] = np.sum(np.abs(scaledquery-scaledtrain))
            elif 'L2' in method:
                res[i] = np.sqrt(np.sum((scaledquery-scaledtrain)**2))
            res[i] /= (scale**2) # normalize over scale
        if all(np.isnan(res)): continue # could not match the feature

        # determine if the min match is acceptable
        res_argmin = np.nanargmin(res)
        scalemin = scalerange[res_argmin]
        if scalemin > 1.2 and res[res_argmin] < 0.8*res[0]:
            expandingKPs.append(tkp)
            scale_argmin.append(scalemin)
            ismatch[idx]=True

            if not showMatches: continue
            
            # recalculate the best matching scaled template
            r = qkp.size*scalemin*1.2/9*20 // 2
            # r = qkp.size*scalemin // 2
            x0,y0 = trunc_coords(trainpatch.shape,(x_tkp-r, y_tkp-r))
            x1,y1 = trunc_coords(trainpatch.shape,(x_tkp+r, y_tkp+r))
            scaledtrain = trainpatch[y0:y1, x0:x1]
            scaledquery = cv2.resize(querypatch, scaledtrain.shape[::-1]
                                     , fx=scalemin, fy=scalemin
                                     , interpolation=cv2.INTER_LINEAR)

            # draw the query patch, the best matching scaled patch and the
            # training patch
            templimg = np.zeros((scaledquery.shape[0]
                                 ,scaledquery.shape[1]+scaledtrain.shape[1]+querypatch.shape[1])
                                , dtype=trainImg.dtype) + 255

            # scale values for display
            querypatch = 255.*(querypatch - np.min(querypatch))/(np.max(querypatch) - np.min(querypatch))
            scaledtrain = 255.*(scaledtrain - np.min(scaledtrain))/(np.max(scaledtrain) - np.min(scaledtrain))
            scaledquery = 255.*(scaledquery - np.min(scaledquery))/(np.max(scaledquery) - np.min(scaledquery))

            drawInto(querypatch, templimg)
            drawInto(scaledquery,templimg,tl=(querypatch.shape[1],0))
            drawInto(scaledtrain,templimg,tl=(querypatch.shape[1]+scaledquery.shape[1],0))
            cv2.drawKeypoints(dispimg,[trainKPs[m.trainIdx]], dispimg, color=(0,0,255)
                              ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            print
            print "scale_range =", repr(scalerange)[6:-1]
            # print "residuals =", repr((res-np.nanmin(res))/(np.nanmax(res)-np.nanmin(res)))[6:-1]
            print "residuals =", repr(res)[6:-1]
            print "Relative scaling of template:",scalemin
            print "Find next match? ('s' to skip remaining matches,'q' to quit,enter or space to continue):",
            sys.stdout.flush()

            global gmain_win
            cv2.imshow(TEMPLATE_WIN, templimg)
            cv2.imshow(gmain_win, dispimg)

            k = cv2.waitKey(100)%256
            while k not in map(ord,('\r','s','q',' ')): k = cv2.waitKey(100)%256
            if k == ord('s'): showMatches=False
            elif k == ord('q'): raise SystemExit

    return expandingKPs, scale_argmin, ismatch, k


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
parser.add_option("--threshold", dest="threshold", type=float, default=1500.
                  , help="Use feed from a ROS bagged recording.")
parser.add_option("-m", "--draw-scale-match", dest="showmatches"
                  , action="store_true", default=False
                  , help="Show scale matches for each expanding keypoint.")
parser.add_option("--no-draw", dest="nodraw"
                  , action="store_true", default=False
                  , help="Draw keypoints and their matches.")
parser.add_option("--video-topic", dest="camtopic", default="/ardrone"
                  , help="Specify the topic for camera feed (default='/ardrone').")
parser.add_option("--video-file", dest="video", default=None
                  , help="Load a video file to test or specify camera ID (typically 0).")
parser.add_option("-r","--record-video", dest="record", default=None
                  , help="Record session to video file.")
parser.add_option("--start", dest="start"
                  , type="int", default=0
                  , help="Starting frame number for video file analysis.")
parser.add_option("--stop", dest="stop"
                  , type="int", default=None
                  , help="Stop frame number for analysis.")
(opts, args) = parser.parse_args()

if opts.bag:
    DEVNULL = open(os.devnull, 'wb')
    bagp = Popen(["rosbag","play",opts.bag],stdout=DEVNULL,stderr=DEVNULL)

video_writer = opts.record

if opts.video:
    try: opts.video = int(opts.video)
    except ValueError: pass
    frmbuf = VideoBuffer(opts.video,opts.start,opts.stop)
else:
    frmbuf = ROSCamBuffer(opts.camtopic+"/image_raw")

# start the node and control loop
rospy.init_node("flownav", anonymous=False)

kbctrl = None
if opts.camtopic == "/ardrone" and not opts.video:
    kbctrl = KeyboardController(max_speed=0.5,cmd_period=100)
if kbctrl:
    FlatTrim = rospy.ServiceProxy("/ardrone/flattrim",Empty())
    Calibrate = rospy.ServiceProxy("/ardrone/imu_recalib",Empty())
else:
    FlatTrim = lambda : None
    Calibrate = lambda : None    

#------------------------------------------------------------------------------#
# Print intro output to user
#------------------------------------------------------------------------------#
print "Options"
print "-"*len("Options")
print "- Subscribed to", (repr(opts.camtopic) if not opts.video else opts.video)
print "- Hessian threshold set at", repr(opts.threshold)
print

print "-"*40
gmain_win = frmbuf.name
cv2.namedWindow(gmain_win, flags=cv2.WINDOW_OPENGL|cv2.WINDOW_NORMAL)
if opts.showmatches: cv2.namedWindow(TEMPLATE_WIN, flags=cv2.WINDOW_OPENGL|cv2.WINDOW_NORMAL)
print "-"*40
print

if kbctrl:
    print "Keyboard Controls for automated controller"
    print "-"*len("Keyboard Controls for automated controller")
    for k,v in CharMap.items():
        print k.ljust(20),'=',repr(v).ljust(5)
    print

print "Additional controls"
print "-"*len("Additional controls")
print "* Press 'q' at any time to quit"
print "* Press 'd' at any time to toggle keypoint drawing"
if opts.video:
    print "* Press 'm' at any time to toggle scale matching drawing"
if kbctrl:
    print "* Press 'f' while drone is landed and level to perform a flat trim"
    print "* Press 'c' when drone is in a stable hover to recalibrate drone's IMU"

#------------------------------------------------------------------------------#
# additional setup before main loop
#------------------------------------------------------------------------------#
# initialize the feature description and matching methods
bfmatcher = cv2.BFMatcher()
surf_ui = cv2.SURF(hessianThreshold=opts.threshold)

# mask out a central portion of the image
lastFrame, t_last = frmbuf.grab()
roi = np.zeros(lastFrame.shape,np.uint8)
scrapY, scrapX = lastFrame.shape[0]//8, lastFrame.shape[1]//16
roi[scrapY:-scrapY, scrapX:-scrapX] = True

if opts.record:
    video_writer = cv2.VideoWriter(opts.record, -1, fps=10,frameSize=lastFrame.shape, isColor=False)

# get keypoints and feature descriptors from query image
queryKP, qdesc = surf_ui.detectAndCompute(lastFrame,roi)

# helper function
getMatchKPs = lambda x: (queryKP[x.queryIdx],trainKP[x.trainIdx])

# generate unique IDs for obstacle matches to track over video
idgen = generateuniqid()
getuniqid = lambda : idgen.next()
obstMatches = OrderedDict()

#------------------------------------------------------------------------------#
# main loop
#------------------------------------------------------------------------------#
errsum = 0
while not rospy.is_shutdown():
    currFrame, t_curr = frmbuf.grab()
    t1 = time.time() # loop timer
    if not currFrame.size: break
    
    # /* Find keypoint matches for this frame and filter them */ #

    # get keypoints and feature descriptors from training image
    trainKP, tdesc = surf_ui.detectAndCompute(currFrame,roi)

    # attempt to adaptively threshold
    # err = len(trainKP)-TARGET_N_KP
    # surf_ui.hessianThreshold += 0.3*(err) + 0.15*(errsum+err)
    # if surf_ui.hessianThreshold < MIN_THRESH: surf_ui.hessianThreshold = MIN_THRESH
    # elif surf_ui.hessianThreshold > MAX_THRESH: surf_ui.hessianThreshold = MAX_THRESH
    # errsum = len(trainKP)-TARGET_N_KP

    # find the best K matches for each keypoint
    if qdesc is None or tdesc is None: matches = []
    else:                              matches = bfmatcher.knnMatch(qdesc,tdesc,k=2)

    # filter out poor matches by ratio test , maximum (descriptor) distance
    matches = [m[0] for m in matches
               if ((len(m)==2 and m[0].distance < 0.6*m[1].distance)
                   or (len(m)==1))
               and m[0].distance < 0.2]

    # filter out matches with outlier spatial distances
    mdist = stats.trim1([diffKP_L2(p0,p1) for p0,p1 in map(getMatchKPs,matches)],0.1)
    if mdist.size:
        threshdist = np.mean(mdist) + 3*np.std(mdist)
        matches = [m for m in matches if diffKP_L2(*getMatchKPs(m)) < threshdist]

    # Draw matches
    dispim = None
    if not opts.nodraw:
        matchKPs = [getMatchKPs(m) for m in matches]
        mkp1, mkp2 = zip(*matchKPs) if matchKPs else ([],[])
        dispim = cv2.drawKeypoints(currFrame,mkp1+mkp2, None, color=(255,0,0))
        map(lambda p: cv2.line(dispim, inttuple(*p[0].pt), inttuple(*p[1].pt), (0,255,0), 2), matchKPs)

    # /* Find expanding keypoints */ #
    cv2.rectangle(dispim,(scrapX,scrapY)
                  ,(currFrame.shape[1]-scrapX,currFrame.shape[0]-scrapY)
                  ,(192,192,192),thickness=2)
    matches = [m for m in matches if trainKP[m.trainIdx].size > queryKP[m.queryIdx].size]
    expandingMatches = estimateKeypointExpansion(lastFrame, currFrame, matches
                                                 , queryKP, trainKP
                                                 , opts.showmatches, dispim)
    expandingKPs, kpscales, ismatch, lastkey = expandingMatches

    # /* Update the 'obstacle' history */ #
    # Add these keypoints to the feature history list
    # found = []
    # for idx,scale in zip(np.where(ismatch)[0],kpscales):
    #      # kpdist = np.array([overlap(kp,trainKP[m.trainIdx]) for clsid,kp in obstMatches.items()],dtype=np.bool)
    #     if queryKP[m.queryIdx].class_id in obstMatches:
    #         class_id = queryKP[m.queryIdx].class_id
    #         trainKP[m.trainIdx].class_id = class_id
    #     # elif np.any(kpdist):
    #     #     class_id = obstMatches.keys()[np.where(kpdist)[0]]
    #     else:
    #         class_id = getuniqid()
    #         obstMatches[class_id] = KeyPoint(trainKP[m.trainIdx])

    #     obstMatches[class_id].detects += 1
    #     obstMatches[class_id].age = -1
    #     obstMatches[class_id].scalehist.append(scale)

    #     obstMatches[class_id].pt = trainKP[m.trainIdx].pt
    #     obstMatches[class_id].size = trainKP[m.trainIdx].size
    #     obstMatches[class_id].response = trainKP[m.trainIdx].response
        
    #     found.append(obstMatches[class_id])
    # for clsid in obstMatches.keys():
    #     obstMatches[clsid].age += 1
    #     if obstMatches[clsid].age > 5: del obstMatches[clsid]

    # Draw a broken vertical line at the estimated obstacle horizontal position
    # newobjsize = sum(kp.size for kp in keypoints)
    # if newobjsize > 300 and newobjsize > oldobjsize:
    #     x_obs, y = avgKP(expandingKPs) if expandingKPs else (currFrame.shape[1]//2,currFrame.shape[0]//2)
    #     cv2.line(dispim, (int(x_obs),scrapY+2), (int(x_obs),currFrame.shape[0]//2-20), (0,255,0), 1)
    #     cv2.line(dispim, (int(x_obs),currFrame.shape[0]//2+20), (int(x_obs),currFrame.shape[0]-scrapY-2), (0,255,0), 1)

    # Draw all expanding key points (unless it was done already)
    if not opts.showmatches or lastkey == ord('s'):
        cv2.drawKeypoints(dispim if dispim is not None else currFrame
                          , expandingKPs, dispim, color=(0,0,255)
                          , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Synthesize detected obstacle information
    timestep = t_curr-t_last
    cluster = ClusterKeypoints(expandingKPs,currFrame)
    t2 = time.time() # end loop timer

    if cluster:
        # c = np.argmax([sum(map(attrgetter('detects'),cl.KPs)) for cl in cluster])
        for c in cluster:
            cv2.rectangle(dispim if dispim is not None else currFrame
                              ,c.p0,c.p1,color=(0,255,255),thickness=2)

    # Print out drone status to the image
    if kbctrl:
        stat = "BATT=%.2f" % (kbctrl.navdata.batteryPercent)
        cv2.putText(dispim,stat,(10,currFrame.shape[0]-10)
                    ,cv2.FONT_HERSHEY_TRIPLEX, 0.65, (0,0,255))

    # /* Handle keyboard events */ #
    # capture key
    cv2.imshow(frmbuf.name, dispim)    
    k = cv2.waitKey(1)%256
    if kbctrl:                  # drone keyboard events
       kbctrl.keyPressEvent(k)
       if k == ord('f'):
           try: FlatTrim()
           except rospy.ServiceException, e: print e
       elif k == ord('c'):
           try: Calibrate()
           except rospy.ServiceException, e: print e
    elif opts.video:            # video file controls
       if lastkey is not None:
           while k not in map(ord,('\r','s','q',' ')): k = cv2.waitKey(100)%256
       if k == ord('m'):
           opts.showmatches ^= True
           if opts.showmatches: cv2.namedWindow(TEMPLATE_WIN,cv2.WINDOW_OPENGL|cv2.WINDOW_NORMAL)
           else:                cv2.destroyWindow(TEMPLATE_WIN)
       while(k == ord('b')):
           frmbuf.seek(-2)
           cv2.imshow(frmbuf.name,frmbuf.grab()[0])
           k = cv2.waitKey(100)%256               
       while(k == ord('f')):
           frmbuf.seek(1)
           cv2.imshow(frmbuf.name,frmbuf.grab()[0])
           k = cv2.waitKey(100)%256
    if k == ord('d'): opts.nodraw ^= True
    if k == ord('q'): break

    t = (time.time()-t1)
    if opts.video and (0.100-t) > 0.001: k = cv2.waitKey(int((0.100-t)*1000))

    # push back our current data
    if opts.record: video_writer.write(dispim)
    lastFrame, queryKP, qdesc, t_last = currFrame, trainKP, tdesc, t_curr
    # sys.stdout.write("Loop time: %8.3f ms, threshold at: %5d, nkeypoints: %4d\r" % (1000.*(t2 - t1), surf_ui.hessianThreshold,len(trainKP)))
    # sys.stdout.flush()

# clean up
if opts.bag: bagp.kill()
if opts.record: video_writer.release()
cv2.destroyAllWindows()
frmbuf.close()
print
