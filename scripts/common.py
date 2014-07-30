inttuple = lambda *x: tuple(map(int,x))

roundtuple = lambda *x: tuple(map(lambda y: int(round(y)),x))


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

