#!/usr/bin/env python
import rospy
from std_srvs.srv import Empty

import cv2
import numpy as np
import scipy.stats as stats

from common import *
from framebuffer import ROSCamBuffer,VideoBuffer
import operator as op
from keyboard_control import KeyboardController,CharMap,KeyMapping
from collections import OrderedDict

import time,sys


TEMPLATE_WIN = "Template matching"
MIN_THRESH = 2000
MAX_THRESH = 2500
LAST_DAY = 5
TARGET_N_KP = 50

gmain_win = "flownav"

VERBOSE = 1

def React(controller,img,timediff):
    v = controller.navdata.vx

    ttc = timediff/relsize
    if ttc <= threshttc:
        if (x_obs-img.shape[1]//2) < 0: self.RollRight()
        if x_obs >= img.shape[1]//2: self.RollLeft()


def ClusterKeypoints(keypoints,kphist,img):
    if len(keypoints) < 2: return []

    cluster = []
    unclusteredKPs = sorted(keypoints,key=op.attrgetter('pt'))
    while unclusteredKPs:
        clust = [unclusteredKPs.pop(0)]
        kp = clust[0]
        i = 0
        while i < len(unclusteredKPs):
            if overlap(kp,unclusteredKPs[i]):
                clust.append(unclusteredKPs.pop(i))
            else:
                i += 1
        if len(clust) > 1 and sum(len(kphist[kp.class_id].scalehist) for kp in clust) > 1:
            cluster.append(Cluster(clust,img))

    return cluster

def MergeClusters(clusters,img):
    mergedclusters = []
    unmergedclusters = sorted(clusters,key=op.attrgetter('pt'))
    while unmergedclusters:
        clust = unmergedclusters.pop(0).KPs
        i = 0
        while i < len(unmergedclusters):
            if bboverlap(cl,unmergedclusters[i]):
                clust += unmergedclusters.pop(i).KPs
            else:
                i += 1
        mergedclusters.append(Cluster(clust,img))

    return mergedclusters


def estimateKeypointExpansion(frmbuf, matches, queryKPs, trainKPs, kphist
                              , showMatches=False, dispimg=None, method='L1'):
    scale_argmin = []
    expandingMatches = []

    scalerange = 1.0+np.arange(0.6+0.0125,step=0.0125)
    res = np.zeros(len(scalerange))
    k = None
    skipMatches = False

    if showMatches:
        tdispim = dispimg.copy() if dispimg is not None else frmbuf.grab(0)[0].copy()

    trainImg = frmbuf.grab(0)[0]
    for m in matches:
        qkp = copyKP(queryKPs[m.queryIdx])
        tkp = trainKPs[m.trainIdx]

        # grab the frame where the keypoint was last detected
        fidx = kphist[qkp.class_id].lastFrameIdx if qkp.class_id in kphist else -1
        queryImg = frmbuf.grab(fidx)[0]

        # /* Extract the query and train image patch and normalize them. */ #
        qkp.size = qkp.size*1.25
        x_qkp,y_qkp = qkp.pt
        r = qkp.size // 2
        x0,y0 = trunc_coords(queryImg.shape,(x_qkp-r, y_qkp-r))
        x1,y1 = trunc_coords(queryImg.shape,(x_qkp+r, y_qkp+r))
        querypatch = queryImg[y0:y1, x0:x1]
        if not querypatch.size: continue
        querypatch = (querypatch-np.mean(querypatch))/np.std(querypatch)

        x_tkp,y_tkp = tkp.pt
        r = qkp.size*scalerange[-1] // 2
        x0,y0 = trunc_coords(trainImg.shape,(x_tkp-r, y_tkp-r))
        x1,y1 = trunc_coords(trainImg.shape,(x_tkp+r, y_tkp+r))
        trainpatch = trainImg[y0:y1, x0:x1]
        if not trainpatch.size: continue
        trainpatch = (trainpatch-np.mean(trainpatch))/np.std(trainpatch)

        # Scale up the query to perform template matching
        x_tkp,y_tkp = x_tkp-x0,y_tkp-y0
        res[:] = np.nan
        for i,scale in enumerate(scalerange):
            r = qkp.size*scale // 2
            x0,y0 = trunc_coords(trainpatch.shape,(x_tkp-r, y_tkp-r))
            x1,y1 = trunc_coords(trainpatch.shape,(x_tkp+r, y_tkp+r))
            scaledtrain = trainpatch[y0:y1, x0:x1]
            if not scaledtrain.size: continue

            scaledquery = cv2.resize(querypatch,scaledtrain.shape[::-1]
                                     , fx=scale,fy=scale
                                     , interpolation=cv2.INTER_LINEAR)

            if method == 'corr':
                res[i] = np.sum(scaledquery*scaledtrain)
            elif method == 'L1':
                res[i] = np.sum(np.abs(scaledquery-scaledtrain))
            elif method == 'L2':
                res[i] = np.sqrt(np.sum((scaledquery-scaledtrain)**2))
            elif method == 'L2sq':
                res[i] = np.sum((scaledquery-scaledtrain)**2)
            res[i] /= (scale**2) # normalize over scale
        if all(np.isnan(res)): continue # could not match the feature

        # determine if the min match is acceptable
        res_argmin = np.nanargmin(res)
        scalemin = scalerange[res_argmin]
        if scalemin > 1.2 and res[res_argmin] < 0.8*res[0]:
            scale_argmin.append(scalemin)
            expandingMatches.append(m)

            if not showMatches: continue

            # recalculate the best matching scaled template
            r = qkp.size*scalemin // 2
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

            cv2.drawKeypoints(tdispim,[tkp], tdispim, color=(0,0,255)
                              ,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            if VERBOSE > 1:
                print
                print "class_id:",qkp.class_id
                print "scale_range =", repr(scalerange)[6:-1]
                # print "residuals =", repr((res-np.nanmin(res))/(np.nanmax(res)-np.nanmin(res)))[6:-1]
                print "residuals =", repr(res)[6:-1]
                print "Number of previous detects:", kphist[qkp.class_id].detects if qkp.class_id in kphist else 0
                # print "Number of previous scale estimates:", len(kphist[qkp.class_id].scalehist) if qkp.class_id in kphist else 1
                print "Frames since last detect:", abs(fidx)
                # print "Frames since first detect:", kphist[qkp.class_id].age+1 if qkp.class_id in kphist else 1
                print "Template size =", querypatch.shape
                print "Relative scaling of template:",scalemin
                print "Find next match? ('s' to skip remaining matches,'q' to quit,enter or space to continue):",
                sys.stdout.flush()

            global gmain_win
            cv2.imshow(TEMPLATE_WIN, templimg)
            cv2.imshow(gmain_win, tdispim)

            k = cv2.waitKey(100)%256
            while k not in map(ord,('\r','s','q',' ')): k = cv2.waitKey(100)%256
            if k == ord('s'): showMatches = False
            elif k == ord('q'): raise SystemExit

    return expandingMatches, scale_argmin, k


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
                  , help="Use feed from a ROS bagged recording. (don't)")

# parser.add_option("-l", "--log", dest="log", default=None
#                   , help="Specify where to output intermediate analysis results. (don't)")

parser.add_option("--threshold", dest="threshold", type=float, default=1500.
                  , help="Set the Hessian threshold for keypoint detection.")

parser.add_option("-m", "--draw-scale-match", dest="showmatches"
                  , action="store_true", default=False
                  , help="Show scale matches for each expanding keypoint.")

parser.add_option("-v", "--verbose", dest="verbose", action="count", default=1
                  , help="Print verbose output to stdout. Multiple v's for more verbosity.")

parser.add_option("-q", "--quiet", dest="quiet", default=False
                  , help="Quiet all output to stdout.")

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

VERBOSE = 0 if opts.quiet else opts.verbose

if opts.bag:
    DEVNULL = open(os.devnull, 'wb')
    bagp = Popen(["rosbag","play",opts.bag],stdout=DEVNULL,stderr=DEVNULL)

# logger = open(opts.log,'wb') if opts.log else None

video_writer = opts.record

if opts.video:
    try: opts.video = int(opts.video)
    except ValueError: pass
    frmbuf = VideoBuffer(opts.video,opts.start,opts.stop,historysize=LAST_DAY+1)
else:
    frmbuf = ROSCamBuffer(opts.camtopic+"/image_raw",historysize=LAST_DAY+1)

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

gmain_win = frmbuf.name
cv2.namedWindow(gmain_win, flags=cv2.WINDOW_OPENGL|cv2.WINDOW_NORMAL)
if opts.showmatches: cv2.namedWindow(TEMPLATE_WIN, flags=cv2.WINDOW_OPENGL|cv2.WINDOW_NORMAL)
print "-"*40
print

#------------------------------------------------------------------------------#
# Print intro output to user
#------------------------------------------------------------------------------#
if VERBOSE:
    print "Options"
    print "-"*len("Options")
    print "- Subscribed to", (repr(opts.camtopic) if not opts.video else opts.video)
    print "- Hessian threshold set at", repr(opts.threshold)
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
surf_ui = cv2.SURF(hessianThreshold=opts.threshold,extended=True)

# mask out a central portion of the image
lastFrame, t_last = frmbuf.grab()
roi = np.zeros(lastFrame.shape,np.uint8)
scrapY, scrapX = lastFrame.shape[0]//4, lastFrame.shape[1]//4
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

trackedKPs = OrderedDict()
trackedDesc = OrderedDict()
keypointHist = OrderedDict()

for kp in queryKP: kp.class_id = getuniqid()

#------------------------------------------------------------------------------#
# main loop
#------------------------------------------------------------------------------#
errsum = 0
while not rospy.is_shutdown():
    currFrame, t_curr = frmbuf.grab()
    t1_loop = time.time() # loop timer
    if not currFrame.size: break

    if VERBOSE > 2: print "Frame time: %8.3f ms" % t_curr

    '''
    First, assign _every_ query keypoint a unique ID
    Note: 1 and -1 are the openCV default class_ids
    '''
    for kp in queryKP:
        if kp.class_id in (1,-1): kp.class_id = getuniqid()

    # # attempt to adaptively threshold
    # err = len(trainKP)-TARGET_N_KP
    # surf_ui.hessianThreshold += 0.3*(err) + 0.05*(errsum+err)
    # if surf_ui.hessianThreshold < MIN_THRESH: surf_ui.hessianThreshold = MIN_THRESH
    # # elif surf_ui.hessianThreshold > MAX_THRESH: surf_ui.hessianThreshold = MAX_THRESH
    # errsum = len(trainKP)-TARGET_N_KP

    '''
    Now, define a one to one mapping to the training keypoints
    '''

    trainKP, tdesc = surf_ui.detectAndCompute(currFrame,roi)

    # Find the best K matches for each keypoint
    if qdesc is None or tdesc is None: matches = []
    else:                              matches = bfmatcher.knnMatch(qdesc,tdesc,k=2)

    # Filter out poor matches by ratio test , maximum (descriptor) distance
    matches = [m[0] for m in matches
               if ((len(m)==2 and m[0].distance < 0.6*m[1].distance) or (len(m)==1))
               and m[0].distance < 0.25]

    for m in matches:
        trainKP[m.trainIdx].class_id = queryKP[m.queryIdx].class_id

    # Filter out matches with outlier spatial distances
    mdist = stats.trim1([diffKP_L2(p0,p1) for p0,p1 in map(getMatchKPs,matches)],0.1)
    if mdist.size:
        threshdist = np.mean(mdist) + 2*np.std(mdist)
        matches = [m for m in matches if diffKP_L2(*getMatchKPs(m)) < threshdist]

    # Draw matches
    dispim = None
    if not opts.nodraw:
        matchKPs = [getMatchKPs(m) for m in matches]
        mkp1, mkp2 = zip(*matchKPs) if matchKPs else ([],[])
        dispim = cv2.drawKeypoints(currFrame,mkp1, None, color=(0,255,0))
        cv2.drawKeypoints(dispim, mkp2, dispim, color=(255,0,0))
        map(lambda p: cv2.line(dispim, inttuple(*p[0].pt), inttuple(*p[1].pt), (0,255,0), 1), matchKPs)
        cv2.rectangle(dispim,(scrapX,scrapY)
                      ,(currFrame.shape[1]-scrapX,currFrame.shape[0]-scrapY)
                      ,(192,192,192),thickness=2)

    # if np.mean(mdist) > 10:
    #     print "Optical flow velocity exceeded"
    #     print np.mean(mdist)
    #     matches = []

    '''
    Find an estimate of the scale change for keypoints that are expanding
    '''
    matches = [m for m in matches if trainKP[m.trainIdx].size > queryKP[m.queryIdx].size]

    matches, kpscales, lastkey = estimateKeypointExpansion(frmbuf, matches
                                                           , queryKP, trainKP, keypointHist
                                                           , opts.showmatches, dispim)
    # update matched expanding keypoints with accurate scale
    # carry over keypoint class ids and update the descriptor
    matchIDs = []
    for m,scale in zip(matches,kpscales):
        clsid = queryKP[m.queryIdx].class_id

        if clsid not in keypointHist:
            keypointHist[clsid] = KeyPointHistory()

        if len(keypointHist[clsid].timehist) != 0:
            t_A = keypointHist[clsid].timehist[-1][-1]
        else:
            t_A = t_last

        # save the keypoint data
        trainKP[m.trainIdx].class_id = clsid
        trackedKPs[clsid] = copyKP(trainKP[m.trainIdx])
        trackedDesc[clsid] = tdesc[m.trainIdx].copy()
        keypointHist[clsid].update(t_A,t_curr,scale)
        matchIDs.append(clsid)
        
    # Update the keypoint history for previously expanding keypoint that were
    # not detected/matched in this frame
    detected = [kp.class_id for kp in trainKP]
    for clsid in keypointHist:
        keypointHist[clsid].age += 1
        keypointHist[clsid].lastFrameIdx -= 1

        if keypointHist[clsid].age == LAST_DAY:                      # you've had your chance
            if VERBOSE > 2: print "Deleting keypoint", clsid            
            del trackedKPs[clsid]
            del trackedDesc[clsid]
            del keypointHist[clsid]
        elif keypointHist[clsid].age > 0 and clsid not in detected: # better luck next time! (add to the next query)
            if VERBOSE > 2: print "Keypoint", clsid, "added to next query."

            trainKP.append(trackedKPs[clsid])
            if tdesc is None:
                tdesc = trackedDesc[clsid].reshape(1,-1)
            else:
                tdesc = np.r_[tdesc,trackedDesc[clsid].reshape(1,-1)]
    t2_loop = time.time() # end loop timer

    # Draw expanding keypoints with tags
    expandingKPs = [trackedKPs[clsid] for clsid in matchIDs]
    if not opts.nodraw:
        cv2.drawKeypoints(dispim, expandingKPs, dispim, color=(0,0,255)
                          , flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        for m in matches:
            qkp = queryKP[m.queryIdx]
            tkp = trainKP[m.trainIdx]
            # c = keypointHist[tkp.class_id].scalehist[-1]
            # a = abs(qkp.pt[1] - (currFrame.shape[1]//2))
            # b = abs(tkp.pt[1] - (currFrame.shape[1]//2))
            scale = np.array(keypointHist[tkp.class_id].scalehist[-1])
            # tstep = np.diff(keypointHist[tkp.class_id].timehist[-1]).flatten() / 1000.
            tstep = 1
            ttc = tstep / (scale - 1)

            kpinfo = "(%d,%.2f,%.3f)" % (keypointHist[tkp.class_id].detects,scale,ttc)
            cv2.putText(dispim,kpinfo,inttuple(tkp.pt[0]+tkp.size//2,tkp.pt[1]-tkp.size//2)
                        ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0))

    '''
    Finally, perform some simple clustering of adjacent keypoints to
    obtain a more accurate estimate of TTC
    '''
    cluster = ClusterKeypoints(expandingKPs,keypointHist,currFrame)
    # cluster = MergeClusters(cluster,currFrame)
    if cluster and not opts.nodraw:
        if VERBOSE:
            print
            print "Clusters found:"
            for c in cluster: print c

        c = np.argmin([max(cl.dist) for cl in cluster])
        c = cluster[c]

        if VERBOSE: print "Cluster flagged as object:", c

        votes = 0
        ttc_hist = []
        for i,kp in enumerate(c.KPs):
            votes += keypointHist[kp.class_id].detects
            # ttc = np.diff(keypointHist[kp.class_id].timehist).flatten() / 1000. / (np.array(keypointHist[kp.class_id].scalehist) - 1)
            ttc = 1 / (np.array(keypointHist[kp.class_id].scalehist) - 1)            
            ttc_hist.append(ttc)

        if VERBOSE > 1:
            disphist = np.zeros((len(ttc_hist),max(len(t) for t in ttc_hist)))
            for i,ttc_row in enumerate(ttc_hist): disphist[i,0:len(ttc_row)] = ttc_row
            print "ttc_hist = np.array(", repr(disphist)[6:-1], ")"

        ttc = np.mean([ttc[ttc != 0][-1] for ttc in ttc_hist])

        clustinfo = "(%d,%d,%.2f)" % (len(c.KPs),votes,ttc)
        if VERBOSE > 1:
            print "Overlapping keypoints:", len(c.KPs)
            print "Votes:", votes
            print "ttc:", ttc
            print 
        cv2.rectangle(dispim,c.p0,c.p1,color=(0,255,255),thickness=2)
        cv2.putText(dispim,clustinfo,(c.p1[0]-5,c.p1[1])
                    ,cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255))        

    # Draw a broken vertical line at the estimated obstacle horizontal position
    # x_obs, y = avgKP(expandingKPs) if expandingKPs else (currFrame.shape[1]//2,currFrame.shape[0]//2)
    # cv2.line(dispim, (int(x_obs),scrapY+2), (int(x_obs),currFrame.shape[0]//2-20), (0,255,0), 1)
    # cv2.line(dispim, (int(x_obs),currFrame.shape[0]//2+20), (int(x_obs),currFrame.shape[0]-scrapY-2), (0,255,0), 1)

    # Print out drone status to the image
    if kbctrl:
        stat = "BATT=%.2f" % (kbctrl.navdata.batteryPercent)
        cv2.putText(dispim,stat,(currFrame.shape[1]-20,currFrame.shape[0]-10)
                    ,cv2.FONT_HERSHEY_TRIPLEX, 0.65, (0,0,255))        
    elif opts.video and frmbuf.stop is not None:
        stat = "FRAME %4d/%4d" % (frmbuf.cap.get(cv2.CAP_PROP_POS_FRAMES),frmbuf.stop)
        cv2.putText(dispim,stat,(10,currFrame.shape[0]-10)
                    ,cv2.FONT_HERSHEY_TRIPLEX, 0.65, (0,0,255))

    ''''
    Handle input keyboard events
    '''
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
           while k not in map(ord,('\r','s','q',' ','m','b','f')): k = cv2.waitKey(250)%256
       if k == ord('m'):
           opts.showmatches ^= True
           if opts.showmatches: cv2.namedWindow(TEMPLATE_WIN,cv2.WINDOW_OPENGL|cv2.WINDOW_NORMAL)
           else:                cv2.destroyWindow(TEMPLATE_WIN)
       while(k == ord('b')):
           frmbuf.seek(-2)
           cv2.imshow(frmbuf.name,frmbuf.grab()[0])
           k = cv2.waitKey(250)%256
       while(k == ord('f')):
           frmbuf.seek(1)
           cv2.imshow(frmbuf.name,frmbuf.grab()[0])
           k = cv2.waitKey(250)%256
    if k == ord('d'): opts.nodraw ^= True
    if k == ord('q'): break

    # limit the display frame rate to 10fps
    t = (time.time()-t1_loop)
    if opts.video and (0.100-t) > 0.001: k = cv2.waitKey(int((0.100-t)*1000))

    if opts.record: video_writer.write(dispim)

    # push back our current data
    lastFrame   = currFrame
    queryKP     = trainKP
    qdesc       = tdesc
    t_last      = t_curr

# clean up
if opts.bag: bagp.kill()
if opts.record: video_writer.release()
cv2.destroyAllWindows()
frmbuf.close()
