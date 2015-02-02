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

BUFSIZE = 50
SCROLLSIZE = 20
NTRACKEDKPS = 10
NMAXLINES = 100

INVALID_INT_VALUE = -1

class Storage(object):
    def __init__(self, a, savepath , chunksize=10):
        self.__index = 0
        self._chunksize = chunksize*len(a)
        self._buffer = np.zeros_like(a)
        self._buffer.resize((self._chunksize,))
        self.path = savepath

    def store(self, a):
        n = len(a)

        if (n+self.__index) > len(self._buffer):
            self._buffer.resize((len(self._buffer)+self._chunksize,))

        self._buffer[self.__index:self.__index+n] = a.copy()
        self.__index += n

    def close(self):
        with open(self.path, 'wb') as storagefile:
            self._buffer.tofile(storagefile)


class DataSubscriber(object):
    def __init__(self, a, savepath=None, scrollsize=SCROLLSIZE):
        self._storage = Storage(a, savepath=savepath) if savepath else None
        self.buffersize = len(a)
        self.scrollsize = scrollsize
        self.returnindex = self.buffersize - self.scrollsize
        self.keypoints = self.prevkeypoints = []

        self._buffer = a
        self._range = np.arange(len(a))

        self.__index = 0
        self.__bufferindex = 1 # start at 1 to allow for indexing the prev frame

        self.subscriber = self.connect()

    def connect(self):
        rospy.init_node('ttcplotter')

        s = rospy.topics.Subscriber("/flownav/data", ttcMsg, self)

        print "Listening for publisher..."
        try:
            while not rospy.core.is_shutdown() and self.__index == 0:
                rospy.rostime.wallsleep(0.01)
        except ROSInterruptException:
            s.unregister()
            raise

        if not rospy.core.is_shutdown(): print "Connected."

        return s

    @property
    def index(self):
        return self.__index
        
    def roll(self):
        if self._storage is not None:
            self._storage.store(self._buffer[:self.scrollsize])

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

        # remove any points that didn't show up this frame
        for l in set(self.prevkeypoints).difference(datum.keypoints[:NTRACKEDKPS]):
            self.keypoints.remove(l)

        dt = datum.timestep.secs+datum.timestep.nsecs/1e9
        for kp in datum.keypoints[:NTRACKEDKPS]:
            clsid = kp.class_id

            if clsid in self.keypoints:
                i = self.keypoints.index(clsid)
            else:
                i = len(self.keypoints)
                self.keypoints.append(clsid)
            obj_size[self.__bufferindex, i] = 1/float(kp.scale-1)
            obj_id[self.__bufferindex, i] = clsid

        # objSize[self.__bufferindex, 0] = np.mean(objSize[self.__bufferindex, 1:len(kpSizes)+1])
        # objSize[self.__bufferindex, 1:len(kpSizes)+1] = kpSizes
        
        self.prevkeypoints = self.keypoints
        self.__bufferindex += 1

    def __exit__(self, exception_type, exception_val, trace):
        self.close()
        if exception_type == rospy.ROSInterruptException:
            print "Process ended."
        return True

    def close(self):
        self.subscriber.unregister()
        if self._storage is not None:
            self._storage.close()
        return True

        
class DataPlotter(DataSubscriber):
    markers = []
    for m in plt.Line2D.markers:
        try:
            if len(m) == 1 and m not in (' ','x'):
                markers.append(m)
        except TypeError:
            pass
    colors=('#a6cee3'
            ,'#1f78b4'
            ,'#b2df8a'
            ,'#33a02c'
            ,'#fb9a99'
            ,'#e31a1c'
            ,'#fdbf6f'
            ,'#ff7f00'
            ,'#cab2d6')

    def __init__(self,*args,**kwargs):
        super(self.__class__,self).__init__(*args,**kwargs)
        self.connections = []

        plt.ion()

        self.fig = plt.figure(figsize=(10.24,7.68),dpi=100,frameon=False)
        ax = self.fig.add_subplot(111)
        ax.set_ylabel("Patch size"); ax.set_xlabel("Frame")
        ax.grid()
        self.fig.tight_layout()

        self.combos = list(product(self.markers,self.colors))
        shuffle(self.combos)
        self.combos = cycle(self.combos)

        self.lastIdx = 0

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
        keypoints = self.keypoints

        if not keypoints: return
        
        axlabels = [line.get_label() for line in ax.lines]
        for i,clsid in enumerate(keypoints):
            if str(clsid) not in axlabels:
                line = self.add_line(label=clsid,alpha=0.75)
            else:
                line = ax.lines.pop(axlabels.index(str(clsid)))
                ax.lines.append(line)
            mask = obj_id[:,i] == clsid
            line.set_data(self._range[mask], obj_size[mask,i])

        ax.relim()
        ax.autoscale_view(scaley=True)
        ax.set_xlim(self._range[0],self._range[-1])

        # Update the legend, put oldest lines first
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(reversed(handles[-NTRACKEDKPS:])
                  , reversed(labels[-NTRACKEDKPS:]))
        self.fig.canvas.draw()

    def close(self):
        super(self.__class__,self).close()
        for cid in self.connections:
            self.fig.canvas.mpl_disconnect(cid)


if __name__ == '__main__':
    buffer_dtype = [('size',np.float64,(NTRACKEDKPS,)), ('id',np.int64,(NTRACKEDKPS,))]
    databuffer = np.zeros((BUFSIZE,),dtype=buffer_dtype)
    databuffer[:] = INVALID_INT_VALUE

    savepath = sys.argv[1] if len(sys.argv)>1 else None

    with DataPlotter(databuffer,savepath=savepath) as p:
        while not rospy.core.is_shutdown():
            p.update_plot()
            plt.pause(0.1)
