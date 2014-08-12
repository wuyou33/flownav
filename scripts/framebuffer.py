from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from ardrone_autonomy.msg import Navdata
import rospy
import cv2
import numpy as np


class VideoBuffer(object):
    def __init__(self,vidfile):
        self.cap = cv2.VideoCapture(vidfile)

    def grab(self):
        img = self.cap.read()[1]
        if img is not None:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            img = np.array([])
        return img

    def close(self):
        self.cap.release()

class ROSCamBuffer(object):
    '''
    ROSCamBuffer

    Creates a subcription node to the image publisher and converts the image
    into opencv image type.
    '''
    def __init__(self, topic):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.imageq = []
        self.image = None
        self.count = 0
        self.msg = None

    def callback(self,data):
        try:
            img = self.bridge.imgmsg_to_cv2(data,'bgr8')
            self.count += 1
            self.msg = data
            self.image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # self.imageq.append(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
        except rospy.ROSException:
            raise
        except KeyboardInterrupt:
            raise

    def grab(self):
        try:
            while self.image is None: None#rospy.sleep(rospy.Duration(secs=0,nsecs=1e3))
        except KeyboardInterrupt:
            raise
        return self.image

    def close(self):
        self.image_sub.unregister()
        del self.imageq
        del self.image
