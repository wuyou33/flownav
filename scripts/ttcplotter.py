#!/usr/bin/env python
import rospy
from flownav.msg import ttc as ttcMsg
from flownav.msg import keypoint as kpMsg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

BUFSIZE = 120
SCROLLSIZE = 30

# plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_ylabel("Relative scale"); ax.set_xlabel("Frame")
# ax.set_xlim(1,BUFSIZE)
ax.set_ylim(0.9,2)
ax.grid()
fig.tight_layout()
graph, = ax.plot([],'b-o')

frames = np.ones(BUFSIZE,dtype=np.uint32)*np.nan
scales = np.ones(BUFSIZE,dtype=np.float64)*np.nan


class Plotter:
    def __init__(self):
        self.__index = 0

    def __call__(self,datum):
        if (self.__index > 2*SCROLLSIZE) and (self.__index % SCROLLSIZE) == 0:
            frames[:-SCROLLSIZE] = frames[SCROLLSIZE:]
            scales[:-SCROLLSIZE] = scales[SCROLLSIZE:]
            frames[-SCROLLSIZE:] = scales[-SCROLLSIZE:] = np.nan

            self.__index = 2*SCROLLSIZE
            ax.set_xlim(np.nanmin(frames),np.nanmax(frames)+SCROLLSIZE)
            ax.autoscale_view(True,True,True)

        frames[self.__index] = datum.frame_id

        if len(datum.keypoints) > 0:
            scales[self.__index] = np.mean([s.scale for s in datum.keypoints])
        
        self.__index += 1
        

def init():
    graph.set_data([],[])
    return graph,

def animate(frameIdx):
    mask = ~np.isnan(scales)
    graph.set_data(frames[mask], scales[mask])
    return graph,

def listener():
    # in ROS, nodes are unique named. If two nodes with the same
    # node are launched, the previous one is kicked off. The 
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaenously.
    rospy.init_node('ttcplotter')
    rospy.Subscriber("/flownav/data", ttcMsg, Plotter())

    # spin() simply keeps python from exiting until this node is stopped
        
if __name__ == '__main__':
    listener()
    anim = animation.FuncAnimation(fig, animate, init_func=init, blit=False, interval=100)
    plt.show()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
