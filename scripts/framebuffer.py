from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from ardrone_autonomy.msg import Navdata
import rospy
import cv2

class FrameBuffer:
    '''
    FrameBuffer

    Creates a subcription node to the image publisher and converts the image
    into opencv image type.
    '''
    def __init__(self, topic="/ardrone/image_raw"):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.imageq = []

    def callback(self,data):
        try:
            img = self.bridge.imgmsg_to_cv2(data,'bgr8')
            self.msg = data
            self.imageq.append(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY))
        except ROSException:
            raise

    def grab(self):
        try:
            while not self.imageq: None
        except KeyboardInterrupt:
            raise
        return self.imageq.pop()
