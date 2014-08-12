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

import time,sys

TEMPLATE_WIN = "Template matching"
MAIN_WIN = "flownav"
MIN_THRESH = 500
MAX_THRESH = 2500
TARGET_N_KP = 100

def React(keypoints,img,controller=None):
    if len(keypoints) < 3: return

    # print "%d obstacles" % len(keypoints)
    # print "Obstacle detected!"
    # print "(ID, scale history, ndetects)"
    # print "\n".join(map(str,map(attrgetter('class_id','age','scalehist','detects'),keypoints)))
    # print np.diff(keypoints[0].scalehist)

    x_obs = avgKP(keypoints)

    if controller:
       if (x_obs-img.shape[1]//2) < 0: self.RollRight()
       if x_obs >= img.shape[1]//2: self.RollLeft()


def estimateKeypointExpansion(queryImg, trainImg, matches, queryKPs, trainKPs, showMatches=False, dispimg=None, method='L1'):
    scalerange = np.r_[1.0,1.0+np.arange(0.2,0.6+0.05,step=0.025)]
    res = np.zeros(len(scalerange))
    expandingKPs = []
    kpscale = []
    matchIdx = []

    for idx,m in enumerate(matches):
        qkp = queryKPs[m.queryIdx]
        tkp = trainKPs[m.trainIdx]

        # /* Extract the query image patch. */ #
        # Use the location of the keypoint in the train image as an estimation
        # of the location in the last frame.
        x_tkp,y_tkp = qkp.pt
        r = qkp.size*1.2/9*20 // 2
        x0,y0 = trunc_coords(queryImg.shape,(x_tkp-r, y_tkp-r))
        x1,y1 = trunc_coords(queryImg.shape,(x_tkp+r, y_tkp+r))
        querypatch = queryImg[y0:y1, x0:x1]
        if not querypatch.size: continue

        x_tkp,y_tkp = tkp.pt
        res[:] = np.nan     # initialize all residuals as invalid
        for i,scale in enumerate(scalerange):
            r = qkp.size*scale*1.2/9*20 // 2
            x0,y0 = trunc_coords(trainImg.shape,(x_tkp-r, y_tkp-r))
            x1,y1 = trunc_coords(trainImg.shape,(x_tkp+r, y_tkp+r))
            traintempl = trainImg[y0:y1, x0:x1]

            if not traintempl.size: break # feature got too large to match

            scaledtempl = cv2.resize(querypatch,traintempl.shape[::-1]
                                     , fx=scale,fy=scale
                                     , interpolation=cv2.INTER_LINEAR)

            # normalize image patches before comparison
            if 'normalized' in method:
                traintempl = (traintempl - np.mean(traintempl))/np.std(traintempl)
                scaledtempl = (scaledtempl - np.mean(scaledtempl))/np.std(scaledtempl)

            # normalize metric with respect to scale
            if 'L1' in method:
                res[i] = np.sum(np.abs(scaledtempl-traintempl))/(scale**2)
            elif 'L2' in method:
                res[i] = np.sqrt(np.sum((scaledtempl-traintempl)**2))/(scale**2)
        if all(np.isnan(res)): continue # could not match the feature

        # determine if this is a solid match
        res_argmin = np.nanargmin(res)
        scalemin = scalerange[res_argmin]
        if scalemin > 1.2 and res[res_argmin] < 0.8*res[~np.isnan(res)][0]:
            expandingKPs.append(trainKPs[m.trainIdx])
            kpscale.append(scalemin)
            matchIdx.append(idx)

            if not showMatches: continue

            # recalculate the best matching scaled template
            r = qkp.size*scalemin*1.2/9*20 // 2
            x0,y0 = trunc_coords(trainImg.shape,(x_tkp-r, y_tkp-r))
            x1,y1 = trunc_coords(trainImg.shape,(x_tkp+r, y_tkp+r))
            traintempl = trainImg[y0:y1, x0:x1]
            scaledtempl = cv2.resize(querypatch, traintempl.shape[::-1]
                                     , fx=scalemin, fy=scalemin
                                     , interpolation=cv2.INTER_LINEAR)

            if 'normalized' in method:
               traintempl = (traintempl - np.mean(traintempl))/np.std(traintempl)            
               scaledtempl = (scaledtempl - np.mean(scaledtempl))/np.std(scaledtempl)

               # scale brightness for display purposes
               scaledtempl *= 255/(np.max(scaledtempl)-np.min(scaledtempl))
               traintempl *= 255/(np.max(traintempl)-np.min(traintempl))            

            # draw the template and the best matching scaled version
            templimg = np.zeros((scaledtempl.shape[0]
                                 ,scaledtempl.shape[1]+traintempl.shape[1]+querypatch.shape[1])
                                , dtype=trainImg.dtype)

            drawInto(querypatch, templimg)
            drawInto(scaledtempl,templimg,tl=(querypatch.shape[1],0))
            drawInto(traintempl,templimg,tl=(querypatch.shape[1]+scaledtempl.shape[1],0))
            dispimg = cv2.drawKeypoints(dispimg,[trainKPs[m.trainIdx]], dispimg, color=(0,0,255)
                                        ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
            cv2.imshow(TEMPLATE_WIN, templimg)
            cv2.imshow(MAIN_WIN, dispimg)
            print 
            print "Relative scaling of template:",scalemin
            print "Find next match? ('s' to skip remaining matches,'q' to quit,enter or space to continue):",
            sys.stdout.flush()

            k = cv2.waitKey(100)%256
            while k not in map(ord,('\r','s','q',' ')):
                k = cv2.waitKey(100)%256
            print
            if k == ord('s'): showMatches=False
            elif k == ord('q'): raise SystemExit

    return expandingKPs, kpscale, matchIdx


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
    frmbuf = VideoBuffer(opts.video)
    if opts.start != 0: frmbuf.cap.set(cv2.CAP_PROP_POS_FRAMES,opts.start)
    # if opts.stop: frmbuf.cap.set(cv2.CAP_PROP_POS_FRAMES,opts.start)
else:           frmbuf = ROSCamBuffer(opts.camtopic+"/image_raw")

# start the node and control loop
rospy.init_node("flownav", anonymous=False)

kbctrl = KeyboardController() if (opts.camtopic == "/ardrone" and not opts.video) else None
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
cv2.namedWindow(MAIN_WIN, flags=cv2.WINDOW_OPENGL|cv2.WINDOW_NORMAL)
if opts.showmatches: cv2.namedWindow(TEMPLATE_WIN, flags=cv2.WINDOW_OPENGL|cv2.WINDOW_NORMAL)
print "-"*40
print

print "Keyboard Controls for automated controller"
print "-"*len("Keyboard Controls for automated controller")
for k,v in CharMap.items():
    print k.ljust(20),'=',repr(v).ljust(5)
print

print "Additional controls"
print "-"*len("Additional controls")
print "* Press 'q' at any time to quit"
print "* Press 'd' at any time to toggle keypoint drawing"
print "* Press 'm' at any time to toggle scale matching drawing"
print "* Press 'f' while drone is landed and level to perform a flat trim"
print "* Press 'c' when drone is in a stable hover to recalibrate drone's IMU"

#------------------------------------------------------------------------------#
# additional setup before main loop
#------------------------------------------------------------------------------#
# initialize the feature description and matching methods
bfmatcher = cv2.BFMatcher()
surf_ui = cv2.SURF(hessianThreshold=opts.threshold)

# mask out a central portion of the image
lastFrame = frmbuf.grab()
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
obstMatches = {}

#------------------------------------------------------------------------------#
# main loop
#------------------------------------------------------------------------------#
errsum = 0
oldobjsize = 0
while not rospy.is_shutdown():
    currFrame = frmbuf.grab()
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
               if len(m)==2
               and m[0].distance < 0.8*m[1].distance
               and m[0].distance < 0.25]

    # filter out matches with outlier spatial distances
    mdist = stats.trim1([diffKP_L2(p0,p1) for p0,p1 in map(getMatchKPs,matches)],0.1)
    if mdist.size:
        threshdist = np.mean(mdist) + 2*np.std(mdist)
        matches = [m for m in matches if diffKP_L2(*getMatchKPs(m)) < threshdist]

    # Draw matches
    dispim = None
    if not opts.nodraw:
        # draw each key point and a line between their match
        matchKPs = [getMatchKPs(m) for m in matches]
        mkp1, mkp2 = zip(*matchKPs) if matchKPs else ([],[])
        dispim = cv2.drawKeypoints(currFrame,mkp1+mkp2, None, color=(255,0,0))
        map(lambda p: cv2.line(dispim, inttuple(*p[0].pt), inttuple(*p[1].pt), (0,255,0), 2), matchKPs)

        # draw the search bounding box
        cv2.rectangle(dispim,(scrapX,scrapY),(currFrame.shape[1]-scrapX,currFrame.shape[0]-scrapY)
                      ,(192,192,192),thickness=2)

    # /* Find expanding keypoints */ #
        
    # select only features that have grown to estimate expansion
    matches = [m for m in matches if trainKP[m.trainIdx].size > queryKP[m.queryIdx].size]

    expandingKPs, kpscales, matchIdx = estimateKeypointExpansion(lastFrame, currFrame, matches
                                                                 , queryKP, trainKP
                                                                 , showMatches=opts.showmatches
                                                                 , dispimg=dispim)

    # /* Update the 'obstacle' history */ #
    # Add these keypoints to the feature history list
    found = []
    for idx,scale in zip(matchIdx,kpscales):
        m = matches[idx]
        if trainKP[m.trainIdx].size < 50: continue # reject small patches
        
        if queryKP[m.queryIdx].class_id in obstMatches:
            trainKP[m.trainIdx].class_id = queryKP[m.queryIdx].class_id
        else:
            # kpdist = np.array([diffKP_L2(kp,trainKP[m.trainIdx]) < 10 for clsid,kp in sorted(obstMatches.items())],dtype=np.bool)
            # if np.any(kpdist):
            #     obstMatches[obstMatches.keys()[np.argmin(kpdist)]].detects += 1
                
            trainKP[m.trainIdx].class_id = getuniqid()
            obstMatches[trainKP[m.trainIdx].class_id] = KeyPoint(trainKP[m.trainIdx])


        obstMatches[trainKP[m.trainIdx].class_id].detects += 1
        obstMatches[trainKP[m.trainIdx].class_id].age = -1
        obstMatches[trainKP[m.trainIdx].class_id].scalehist.append(scale)
        
        found.append(obstMatches[trainKP[m.trainIdx].class_id])
    for clsid in obstMatches.keys():
        obstMatches[clsid].age += 1
        if obstMatches[clsid].age > 5: del obstMatches[clsid]

    # Synthesize detected obstacle information
    React(obstMatches.values(),currFrame,controller=kbctrl)
    t2 = time.time() # end loop timer

    # Draw all expanding key points (unless it was done already)
    if not opts.showmatches:
        dispim = cv2.drawKeypoints(dispim if dispim is not None else currFrame
                                   , expandingKPs, dispim, color=(0,0,255)
                                   , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Print out drone status to the image
    if kbctrl:
        stat = "BATT=%.2f" % (kbctrl.navdata.batteryPercent)
        cv2.putText(dispim,stat,(10,currFrame.shape[0]-10)
                    ,cv2.FONT_HERSHEY_TRIPLEX, 0.65, (0,0,255))

    # Draw a broken vertical line at the estimated obstacle horizontal position
    newobjsize = sum(kp.size for kp in expandingKPs)
    if newobjsize > 300 and newobjsize > oldobjsize:
        # print newobjsize
        # print newobjsize - oldobjsize
        x_obs, y = avgKP(expandingKPs) if expandingKPs else (currFrame.shape[1]//2,currFrame.shape[0]//2)
        cv2.line(dispim, (int(x_obs),scrapY+2), (int(x_obs),currFrame.shape[0]//2-20), (0,255,0), 1)
        cv2.line(dispim, (int(x_obs),currFrame.shape[0]//2+20), (int(x_obs),currFrame.shape[0]-scrapY-2), (0,255,0), 1) 

    if opts.record: video_writer.write(dispim)
    cv2.imshow(MAIN_WIN, dispim)

    # delay frame rate if it is faster than 10fps
    t = (time.time()-t1)
    if (0.100-t) > 0.001 : k = cv2.waitKey(int((0.100-t)*1000))
    else: k = cv2.waitKey(1)%256

    # Handle keyboard events
    if kbctrl:
        kbctrl.keyPressEvent(k)
    if k == ord('m'):
        opts.showmatches ^= True
        if opts.showmatches: cv2.namedWindow(TEMPLATE_WIN, flags=cv2.WINDOW_OPENGL|cv2.WINDOW_NORMAL)
        else: cv2.destroyWindow(TEMPLATE_WIN)
    elif k == ord('d'):
        opts.nodraw ^= True
    elif k == ord('f'):
        try: FlatTrim()
        except rospy.ServiceException, e: print e
    elif k == ord('c'):
        try: Calibrate()
        except rospy.ServiceException, e: print e
    elif k == ord('q'):
        break

    lastFrame, queryKP, qdesc = currFrame, trainKP, tdesc
    oldobjsize = newobjsize
    # sys.stdout.write("Loop time: %8.3f ms, threshold at: %5d, nkeypoints: %4d\r" % (1000.*(t2 - t1), surf_ui.hessianThreshold,len(trainKP)))
    sys.stdout.flush()    

# clean up
if opts.bag: bagp.kill()
if opts.record: video_writer.release()
cv2.destroyAllWindows()
frmbuf.close()
print

