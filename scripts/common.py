import numpy as np
import cv2
from operator import attrgetter


class KeyPoint(object):
    def __init__(self,kp=None):
        self.detects = 0
        self.age = 0
        self.scalehist = []
        self.timehist = []

        self.class_id = -1
        self.angle = 0
        self.octave = 0
        self.pt = (0,0)
        self.response = 0
        self.size = 0

        if kp: copyKP(kp,self)


class Cluster(object):
    def __init__(self,keypoints,img):
        self.mask = np.zeros_like(img)
        for kp in keypoints:
            cv2.circle(self.mask,inttuple(*kp.pt),int(kp.size//2),1,thickness=-1)
        self.area = np.sum(self.mask)
        self.pt = findCoM(self.mask)
        self.p0, self.p1 = BlobBoundingBox(self.mask)
        self.KPs = [KeyPoint(kp) for kp in keypoints]
        self.cluster_id = -1
        self.votes = sum(map(attrgetter('detects'),self.KPs))
        self.detects = np.median(map(attrgetter('detects'),self.KPs))
        # dist = []
        # for i in range(len(self.KPs)-1):
        #     j = i + 1
        #     while j < len(self.KPs):
        dist = [diffKP_L2(self.KPs[i],self.KPs[j]) for i in range(len(self.KPs)-1) for j in range(i+1,len(self.KPs))]
                # j += 1
        self.density = min(dist)/max(dist)

    def __repr__(self):
        return str(map(repr,(self.pt,self.area,len(self.KPs))))


def BlobBoundingBox(blob):
    diff = blob.any(axis=0)
    ones = np.flatnonzero(diff)
    xmin, xmax = ones[0], ones[-1]

    diff = blob.any(axis=1)
    ones = np.flatnonzero(diff)
    ymin, ymax = ones[0], ones[-1]

    return (xmin,ymin),(xmax,ymax)


def findCoM(mask):
    colnums = np.arange(np.shape(mask)[1]).reshape(1,-1)
    rownums = np.arange(np.shape(mask)[0]).reshape(-1,1)

    x = np.sum(mask*colnums) // np.sum(mask)
    y = np.sum(mask*rownums) // np.sum(mask)

    return x, y


trunc_coords = lambda shape,xy: [x if x >= 0 and x <= dimsz else (0 if x < 0 else dimsz)
                                 for dimsz,x in zip(shape[::-1],xy)]

bboverlap = lambda cl1,cl2: (cl1.p0[0] <= cl2.p1[0] and cl1.p1[0] >= cl2.p0[0]) and (cl1.p0[1] <= cl2.p1[1] and cl1.p1[1] >= cl2.p0[1])

overlap = lambda kp1,kp2: (kp1.size//2+kp2.size//2) > diffKP_L2(kp1,kp2)

diffKP_L2 = lambda kp0,kp1: np.sqrt((kp0.pt[0]-kp1.pt[0])**2 + (kp0.pt[1]-kp1.pt[1])**2)

diffKP = lambda kp0,kp1: (kp0.pt[0]-kp1.pt[0], kp0.pt[1]-kp1.pt[1])

difftuple_L2 = lambda p0,p1: np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)

difftuple = lambda p0,p1: (p1[0]-p0[0],p1[1]-p0[1])

inttuple = lambda *x: tuple(map(int,x))

roundtuple = lambda *x: tuple(map(int,map(round,x)))

avgKP = lambda keypoints: map(lambda x: sum(x)/len(keypoints),zip(*map(attrgetter('pt'),keypoints)))

toKeyPoint_cv = lambda kp: cv2.KeyPoint(kp.pt[0],kp.pt[1],kp.size,_angle=kp.angle,_response=kp.response,_octave=kp.octave,_class_id=kp.class_id)

def reprObj(obj):
    return "\n".join(["%s = %s" % (attr, getattr(obj, attr)) for attr in dir(obj) if not attr.startswith('_') and not callable(getattr(src,attr))])

def cvtIdx(pt,shape):
    return int(pt[1]*shape[1] + pt[0]) if hasattr(pt, '__len__') else map(int, (pt%shape[1], pt//shape[1]))


def drawInto(src, dst, tl=(0,0)):
    dst[tl[1]:tl[1]+src.shape[0], tl[0]:tl[0]+src.shape[1]] = src

def copyKP(src,dst=None):
    if dst is None: dst = KeyPoint()
    for attr in dir(src):
        if not attr.startswith('_') and not callable(getattr(src,attr)):
            setattr(dst,attr,getattr(src,attr))
    return dst
