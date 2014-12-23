import cv2
import numpy as np
from common import *

VERBOSE = 0
TEMPLATE_WIN = None
MAIN_WIN = None
KEYPOINT_SCALE = 1.2/9*20
SEARCH_RES = 20
scalerange = 1 + (np.arange(11)/float(SEARCH_RES))

def drawTemplateMatches(frmbuf,matches,queryKPs,trainKPs,kphist,scales,dispim=None):
    tdispim = dispim.copy() if dispim is not None else frmbuf.grab(0)[0].copy()

    k = None
    trainImg = frmbuf.grab(0)[0]
    for m,scale in zip(matches,scales):
        qkp = copyKP(queryKPs[m.queryIdx])
        tkp = trainKPs[m.trainIdx]
        qkp.size *= KEYPOINT_SCALE

        # grab the frame where the keypoint was last detected
        fidx = kphist[qkp.class_id].lastFrameIdx if qkp.class_id in kphist else -1
        queryImg = frmbuf.grab(fidx)[0]

        # /* Extract the query and train image patch and normalize them. */ #
        x_qkp,y_qkp = qkp.pt
        r = qkp.size // 2
        x0,y0 = trunc_coords(queryImg.shape,(x_qkp-r, y_qkp-r))
        x1,y1 = trunc_coords(queryImg.shape,(x_qkp+r, y_qkp+r))
        querypatch = queryImg[y0:y1, x0:x1]
        querypatch = (querypatch-np.mean(querypatch))/np.std(querypatch)

        x_tkp,y_tkp = tkp.pt
        r = qkp.size*scalerange[-1] // 2        
        x0,y0 = trunc_coords(trainImg.shape,(x_tkp-r, y_tkp-r))
        x1,y1 = trunc_coords(trainImg.shape,(x_tkp+r, y_tkp+r))
        trainpatch = trainImg[y0:y1, x0:x1]
        trainpatch = (trainpatch-np.mean(trainpatch))/np.std(trainpatch)

        # recalculate the best matching scaled template
        r = qkp.size*scale // 2
        x_tkp,y_tkp = x_tkp-x0,y_tkp-y0
        x0,y0 = trunc_coords(trainpatch.shape,(x_tkp-r, y_tkp-r))
        x1,y1 = trunc_coords(trainpatch.shape,(x_tkp+r, y_tkp+r))
        scaledtrain = trainpatch[y0:y1, x0:x1]
        scaledquery = cv2.resize(querypatch, scaledtrain.shape[::-1]
                                 , fx=scale, fy=scale
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

        cv2.imshow(TEMPLATE_WIN, templimg)
        cv2.imshow(MAIN_WIN, tdispim)

        if VERBOSE:
            print "Find next match? ('s' to skip remaining matches,'q' to quit,enter or space to continue):",

        k = cv2.waitKey(100)%256
        while k not in map(ord,('\r','s','q',' ','m')):
            k = cv2.waitKey(100)%256
        if VERBOSE: print "\r"

        if k in (ord('s'),ord('q'),ord('m')):
            if VERBOSE: print
            return k

    return k


def estimateKeypointExpansion(frmbuf, matches, queryKPs, trainKPs, kphist, method='L2sq'):
    scale_argmin = []
    expandingMatches = []

    trainImg = frmbuf.grab(0)[0]
    res = np.zeros(len(scalerange))
    for m in matches:
        qkp = copyKP(queryKPs[m.queryIdx])
        tkp = trainKPs[m.trainIdx]
        qkp.size *= KEYPOINT_SCALE

        # grab the frame where the keypoint was last detected
        fidx = kphist[qkp.class_id].lastFrameIdx if qkp.class_id in kphist else -1
        queryImg = frmbuf.grab(fidx)[0]

        # /* Extract the query and train image patch and normalize them. */ #
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
        res[:] = np.nan
        x_tkp,y_tkp = x_tkp-x0,y_tkp-y0
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

            if VERBOSE > 2:
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

    return expandingMatches, scale_argmin
