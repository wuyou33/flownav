import numpy as np
import cv2

MAX_AGE = 10

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
        if kp: copyKP(kp,self)

    # def __eq__(self,rhs):
    #     return self.class_id == rhs.class_id

    # def __lt__(self,rhs):
    #     return self.age < rhs.age

    # def __gt__(self,rhs):
    #     return self.age > rhs.age

inimage = lambda shape,xy: map(lambda sx: sx[1] if sx[1] >= 0 and sx[1] <= sx[0] else (0 if sx[1] < 0 else sx[0]), zip(shape[::-1],xy))

diffKP_L2 = lambda kp0,kp1: np.sqrt((kp0.pt[0]-kp1.pt[0])**2 + (kp0.pt[1]-kp1.pt[1])**2)
diffKP = lambda kp0,kp1: (kp0.pt[0]-kp1.pt[0], kp0.pt[1]-kp1.pt[1])

difftuple = lambda p0,p1: np.sqrt((p0[0]-p1[0])**2 + (p0[1]-p1[1])**2)

inttuple = lambda *x: tuple(map(int,x))

roundtuple = lambda *x: tuple(map(int,map(round,x)))

avgtuple = lambda x: map(lambda y: sum(y)/len(y),zip(*x))

def reprObj(obj):
    return "\n".join(["%s = %s" % (attr, getattr(obj, attr)) for attr in dir(obj) if not attr.startswith('_')])

def cvtIdx(pt,shape):
    return int(pt[1]*shape[1] + pt[0]) if hasattr(pt, '__len__') else map(int, pt%shape[1], pt//shape[1])


def drawInto(src, dst, tl=(0,0)):
    dst[tl[1]:tl[1]+src.shape[0], tl[0]:tl[0]+src.shape[1]] = src

def copyKP(src,dst=None):
    if dst is None: dst = cv2.KeyPoint()
    for attr in dir(src):
        if not attr.startswith('_'): setattr(dst,attr,getattr(src,attr))
    return dst
