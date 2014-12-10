#!/usr/bin/env python
import sys
import rospy
from flownav.msg import ttc as ttcMsg
from flownav.msg import keypoint as kpMsg
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib import animation

BUFSIZE = 160
SCROLLSIZE = 15
NTRACKEDKPS = 10
AUTOSCALE = True

buffer_dtype = [('size',np.uint32,(NTRACKEDKPS,)), ('id',np.uint32,(NTRACKEDKPS,))]
databuffer = np.zeros((BUFSIZE,),dtype=buffer_dtype)
frames = np.arange(BUFSIZE,dtype=np.uint64)
databuffer.fill(np.nan)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel("Relative scale"); ax.set_xlabel("Frame")
# ax.set_xlim(1,BUFSIZE)
# ax.set_ylim(0.9,2)
ax.grid()
fig.tight_layout()

markers = []
for m in plt.Line2D.markers:
    try:
        if len(m) == 1 and m != ' ':
            markers.append(m)
    except TypeError:
        pass
colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')

lines = [plt.Line2D((),(),marker=markers[i],color=colors[i%len(colors)])
         for i in range(databuffer['size'].shape[1])]
lines[0].set_linewidth(2)
lines[0].set_alpha(0.6)

legend = plt.legend([plt.Line2D([],[], linewidth=5, color=c) for c in colors]
                    , [], prop={'size':'medium'}, frameon=False,ncol=len(colors)
                    , bbox_to_anchor=(0, 1.015, 1., .05), mode='expand')
ax.add_artist(legend)


class Storage:
    def __init__(self, a, path, chunkSize=10):
        self.__index = 0
        self._chunkSize = chunkSize*len(a)
        self._buffer = np.zeros_like(a)
        self._buffer.resize((self._chunkSize,))
        self.path = path

    def store(self, a):
        n = len(a)

        if n+self.__index > len(self._buffer):
            self._buffer.resize((len(self._buffer)+self._chunkSize,))

        self._buffer[self.__index:self.__index+n] = a.copy()
        self.__index += n

    def close(self):
        with open(self.path, 'wb') as storagefile:
            self._buffer.tofile(storagefile)


class Plotter:
    def __init__(self, a, storeData=False, dataPath='arr.npy', scrollSize=SCROLLSIZE):
        self.__index = 0
        self.__startFrame = 0
        self.storage = Storage(a, dataPath) if storeData else None
        self._buffer = a
        self.bufSize = len(a)
        self._indexBuffer = np.arange(len(a))
        self.scrollSize = scrollSize

    def roll(self):
        global frames

        if self.storage is not None:
            self.storage.store(self._buffer[:self.scrollSize])

        frames += self.scrollSize
        self._buffer[:-self.scrollSize] = self._buffer[self.scrollSize:]
        self._buffer[-self.scrollSize:] = np.nan        
        self.__index = self.bufSize-self.scrollSize-1

    def __call__(self, datum):
        objSize = self._buffer['size']

        if self.__index > 0 and self.__index == self.bufSize:
            self.roll()

            global AUTOSCALE, frames
            if AUTOSCALE and np.nanmin(objSize) != np.nanmax(objSize):
                ax.set_xlim(frames[0],frames[0]+self.bufSize-self.scrollSize)
                ax.set_ylim(np.nanmin(objSize),np.nanmax(objSize))
                ax.autoscale_view(True,True,True)

        keypoints = datum.keypoints
        kpSizes = [keypoints[i].trainSize for i in range(min(NTRACKEDKPS,len(keypoints)))]
                   # for i in range(min(objSize.shape[0]-1,len(datum.keypoints)))]
        ids = [keypoints[i].class_id for i in range(min(NTRACKEDKPS,len(keypoints)))]

        if len(kpSizes) > 0:
            self._buffer['id'][self.__index, :len(kpSizes)] = ids
            objSize[self.__index, :len(kpSizes)] = kpSizes
            # objSize[self.__index, 0] = np.mean(kpSizes)
            # objSize[self.__index, 1:len(kpSizes)+1] = kpSizes
        
        self.__index += 1

    def close(self):
        if self.storage is not None:
            self.storage.close()
        

def init():
    global legend, lines, ax
    for graph in lines:
        ax.add_line(graph)
    ax.hold(False)
    return lines, legend

def animate(frameIdx):
    global lines, legend, databuffer
    mask = ~np.isnan(databuffer['size'][:-SCROLLSIZE,0])

    for i,graph in enumerate(lines):
        data = databuffer['size'][mask,i]
        if not data.size:
            ax.lines.remove(graph)
            continue
        if graph not in ax.lines:
            ax.add_line(graph)
        graph.set_data(frames[mask], data)

    return lines, legend, ax

def listener(callback):
    # in ROS, nodes are unique named. If two nodes with the same
    # node are launched, the previous one is kicked off. The 
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaenously.
    sys.stdout.write("Attempting to connect to publisher...")
    rospy.init_node('ttcplotter')
    rospy.Subscriber("/flownav/data", ttcMsg, callback)
    print "Connected."
    # spin() simply keeps python from exiting until this node is stopped


if __name__ == '__main__':
    p = Plotter(databuffer,storeData=False, dataPath='../data/arr.npy')

    listener(p)
    anim = animation.FuncAnimation(fig, animate, init_func=init, blit=False, interval=100)

    plt.show()
    p.close()
