inimage = lambda shape,xy: map(lambda sx: sx[1] if sx[1] >= 0 and sx[1] <= sx[0] else (0 if sx[1] < 0 else sx[0]), zip(shape[::-1],xy))

difftuple = lambda p0,p1: abs(p0[0]-p1[0]) + abs(p0[1]-p1[1])

inttuple = lambda *x: (int(x[0]),int(x[1]))

roundtuple = lambda *x: (int(round(x[0])),int(round(x[1])))


def reprObj(obj):
    return "\n".join(["%s = %s" % (attr, getattr(obj, attr)) for attr in dir(obj) if not attr.startswith('_')])


def drawMatch(dm,kp1,kp2,dispim,color=-1,thickness=2):
    cv2.line(dispim
             ,tuple(map(int,kp1[dm.queryIdx].pt))
             ,tuple(map(int,kp2[dm.trainIdx].pt)),color,thickness)


def cvtIdx(pt,shape):
    return int(pt[1]*shape[1] + pt[0]) if hasattr(pt, '__len__') else map(int, pt%shape[1], pt//shape[1])


def drawInto(src, dst, tl=(0,0)):
    x, y = tl
    dst[y:y+src.shape[0], x:x+src.shape[1]] = src    

