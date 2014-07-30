from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
from sensor_msgs.msg import Image
import rospy
import cv2

class FrameBuffer:
    '''
    FrameBuffer

    Creates a subcription node to the image publisher and converts the image
    into opencv image type.
    '''
    def __init__(self, topic="/image_raw", node="Image2cv"):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.image = None
        self.grabbed = True
        rospy.init_node(node, anonymous=False)

    def callback(self,data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(data,'bgr8')
            self.image = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
            self.grabbed = False
        except:
            raise

    def grab(self):
        try:
            while self.grabbed is True: None
            self.grabbed = True
        except KeyboardInterrupt:
            raise
        return self.image
