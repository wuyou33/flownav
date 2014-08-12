import numpy as np
import cv2
from operator import attrgetter


class KeyPoint(object):
    def __init__(self,kp=None):
        self.detects = 0
        self.age = 0
        self.class_id = -1
        self.angle = 0
        self.octave = 0
        self.pt = (0,0)
        self.response = 0
        self.size = 0
        self.scalehist = []
        if kp: copyKP(kp,self)


def BlobBoundingBox(blob):
    diff = blob.any(axis=0)
    ones = np.flatnonzero(diff)
    xmin, xmax = ones[0], ones[-1]

    diff = blob.any(axis=1)
    ones = np.flatnonzero(diff)
    ymin, ymax = ones[0], ones[-1]

    return (xmin,ymin),(xmax,ymax)


trunc_coords = lambda shape,xy: [x if x >= 0 and x <= dimsz else (0 if x < 0 else dimsz)
                                 for dimsz,x in zip(shape[::-1],xy)]

diffKP_L2 = lambda kp0,kp1: np.sqrt((kp0.pt[0]-kp1.pt[0])**2 + (kp0.pt[1]-kp1.pt[1])**2)

diffKP = lambda kp0,kp1: (kp0.pt[0]-kp1.pt[0], kp0.pt[1]-kp1.pt[1])

difftuple = lambda p0,p1: np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)

inttuple = lambda *x: tuple(map(int,x))

roundtuple = lambda *x: tuple(map(int,map(round,x)))

avgKP = lambda keypoints: map(lambda x: sum(x)/len(keypoints),zip(*map(attrgetter('pt'),keypoints)))

def reprObj(obj):
    return "\n".join(["%s = %s" % (attr, getattr(obj, attr)) for attr in dir(obj) if not attr.startswith('_')])

def cvtIdx(pt,shape):
    return int(pt[1]*shape[1] + pt[0]) if hasattr(pt, '__len__') else map(int, pt%shape[1], pt//shape[1])


def drawInto(src, dst, tl=(0,0)):
    dst[tl[1]:tl[1]+src.shape[0], tl[0]:tl[0]+src.shape[1]] = src

def copyKP(src,dst=None):
    if dst is None: dst = KeyPoint()
    for attr in dir(src):
        if not attr.startswith('_'): setattr(dst,attr,getattr(src,attr))
    return dst
