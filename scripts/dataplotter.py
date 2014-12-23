#!/usr/bin/env python
import sys
import rospy
from flownav.msg import ttc as ttcMsg
from flownav.msg import keypoint as kpMsg
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from itertools import cycle, product
from random import shuffle

BUFSIZE = 160
SCROLLSIZE = 80
NTRACKEDKPS = 10
NMAXLINES = 100

INVALID_INT_VALUE = -1

class Storage:
    def __init__(self, a, path, chunkSize=10):
        self.__index = 0
        self._chunkSize = chunkSize*len(a)
        self._buffer = np.zeros_like(a)
        self._buffer.resize((self._chunkSize,))
        self.path = path

    def store(self, a):
        n = len(a)

        if (n+self.__index) > len(self._buffer):
            self._buffer.resize((len(self._buffer)+self._chunkSize,))

        self._buffer[self.__index:self.__index+n] = a.copy()
        self.__index += n

    def close(self):
        with open(self.path, 'wb') as storagefile:
            self._buffer.tofile(storagefile)


class DataSubscriber(object):
    def __init__(self, a, storeData=False, dataPath='arr.npy', scrollsize=SCROLLSIZE):
        self.storage = Storage(a, dataPath) if storeData else None
        self.buffersize = len(a)
        self.scrollsize = scrollsize
        self.returnindex = self.buffersize - self.scrollsize
        self.lines = self.lastlines = []
        self.connections = []

        self._buffer = a
        self._range = np.arange(len(a))

        self.__index = 0
        self.__bufferindex = 1 # start at 1 to allow for indexing the prev frame

        self.subscriber = self.connect()

    def connect(self):
        print "Attempting to connect to publisher..."

        rospy.init_node('ttcplotter')

        try:
            s = rospy.topics.Subscriber("/flownav/data", ttcMsg, self)
            while not rospy.core.is_shutdown() and self.__index == 0:
                rospy.rostime.wallsleep(0.01)
            if not rospy.core.is_shutdown(): print "Connected."
        except ROSInterruptException:
            s.unregister()
            raise

        return s

    @property
    def index(self): return self.__index
        
    def roll(self):
        if self.storage is not None:
            self.storage.store(self._buffer[:self.scrollsize])

        self._range += self.scrollsize
        self._buffer[:-self.scrollsize] = self._buffer[self.scrollsize:]
        self._buffer[-self.scrollsize:] = INVALID_INT_VALUE

        self.__bufferindex = self.returnindex-1

    def __enter__(self):
        return self

    def __call__(self, datum):
        self.__index += 1
        if self.__bufferindex == self.buffersize: self.roll()

        obj_size = self._buffer['size']
        obj_id = self._buffer['id']

        for l in set(self.lastlines).difference(datum.keypoints[:NTRACKEDKPS]):
            self.lines.remove(l)

        for kp in datum.keypoints[:NTRACKEDKPS]:
            clsid = kp.class_id

            if clsid in self.lines:
                i = self.lines.index(clsid)
            else:
                i = len(self.lines)
                self.lines.append(clsid)
                obj_size[self.__bufferindex-1, i] = kp.querySize
                obj_id[self.__bufferindex-1, i] = clsid

            obj_size[self.__bufferindex, i] = kp.trainSize
            obj_id[self.__bufferindex, i] = clsid

        # objSize[self.__bufferindex, 0] = np.mean(kpSizes)
        # objSize[self.__bufferindex, 1:len(kpSizes)+1] = kpSizes
        
        self.lastlines = self.lines
        self.__bufferindex += 1

    def __exit__(self, exception_type, exception_val, trace):
        self.close()
        if exception_type == rospy.ROSInterruptException:
            print "Process ended."
        return True

    def close(self):
        self.subscriber.unregister()
        if self.storage is not None: self.storage.close()
        for cid in self.connections:
            self.fig.canvas.mpl_disconnect(cid)
        return True

        
class DataPlotter(DataSubscriber):
    markers = []
    for m in plt.Line2D.markers:
        try:
            if len(m) == 1 and m != ' ':
                markers.append(m)
        except TypeError:
            pass
    # colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k')
    colors=('#a6cee3'
            ,'#1f78b4'
            ,'#b2df8a'
            ,'#33a02c'
            ,'#fb9a99'
            ,'#e31a1c'
            ,'#fdbf6f'
            ,'#ff7f00'
            ,'#cab2d6')

    def __init__(self,a,**kwargs):
        super(self.__class__, self).__init__(a,**kwargs)

        plt.ion()
        self.fig = plt.figure(figsize=(10.24,7.68),dpi=100,frameon=False)
        ax = self.fig.add_subplot(111)
        ax.set_ylabel("Patch size"); ax.set_xlabel("Frame")
        ax.grid()
        self.fig.tight_layout()
        self.combos = list(product(self.markers,self.colors))
        shuffle(self.combos)
        self.combos = cycle(self.combos)

        cid = self.fig.canvas.mpl_connect('resize_event', self.resize)
        self.connections.append(cid)

        self.lastIdx = 0

    def resize(self,event):
        print "Yeah!"
        self.fig.canvas.SetSizeWH(event.width,event.height)

    def add_line(self,**kwargs):
        marker, color = self.combos.next()
        ax = self.fig.axes[0]
        if len(ax.lines) == NMAXLINES: ax.lines[0].remove()

        line = plt.Line2D((),(),marker=marker,color=color,**kwargs)
        return self.fig.axes[0].add_line(line)

    def update_plot(self):
        if self.lastIdx == self.index: return
        self.lastIdx = self.index

        ax = self.fig.axes[0]
        obj_size = self._buffer['size']
        obj_id = self._buffer['id']
        lines = self.lines

        if not lines: return
        
        axlabels = [line.get_label() for line in ax.lines]
        for i,line_id in enumerate(lines):
            if str(line_id) not in axlabels:
                line = self.add_line(label=line_id,alpha=0.75)
            else:
                line = ax.lines.pop(axlabels.index(str(line_id)))
                ax.lines.append(line)
            mask = obj_id[:,i] == line_id
            line.set_data(self._range[mask], obj_size[mask,i])

        ax.relim()
        ax.autoscale_view(scaley=True)
        ax.set_xlim(self._range[0],self._range[-1])

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles[-NTRACKEDKPS:])
                  , reversed(labels[-NTRACKEDKPS:]))
        self.fig.canvas.draw()


if __name__ == '__main__':
    buffer_dtype = [('size',np.int64,(NTRACKEDKPS,)), ('id',np.int64,(NTRACKEDKPS,))]
    databuffer = np.zeros((BUFSIZE,),dtype=buffer_dtype)
    databuffer[:] = INVALID_INT_VALUE

    dataPath = sys.argv[1] if len(sys.argv)>1 else None
    storeData = len(sys.argv)>1

    with DataPlotter(databuffer,storeData=storeData,dataPath=dataPath) as p:
        while not rospy.core.is_shutdown():
            p.update_plot()
            plt.pause(0.1)
