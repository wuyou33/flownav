import rospy
from geometry_msgs.msg import Twist  	 # for sending commands to the drone
from std_msgs.msg import Empty       	 # for land/takeoff/emergency
from ardrone_autonomy.msg import Navdata # for receiving navdata feedback
from operator import itemgetter

# Some Constants
COMMAND_PERIOD = 100 #ms

DroneStatus = dict(Emergency = 0
                   ,Inited = 1
                   ,Landed = 2
                   ,Flying = 3
                   ,Hovering = 4
                   ,Test = 5
                   ,TakingOff = 6
                   ,GotoHover = 7
                   ,Landing = 8
                   ,Looping = 9)


class DroneController(object):
    def __init__(self):
        self.subNavdata = rospy.Subscriber('/ardrone/navdata',Navdata,self.ReceiveNavdata)
        self.navdata=Navdata()

        self.pubLand    = rospy.Publisher('/ardrone/land',Empty,)
        self.pubTakeoff = rospy.Publisher('/ardrone/takeoff',Empty)
        self.pubReset   = rospy.Publisher('/ardrone/reset',Empty)

        self.pubCommand = rospy.Publisher('/cmd_vel',Twist)

        self.command = Twist()
        self.commandTimer = rospy.Timer(rospy.Duration(COMMAND_PERIOD/1000.0),self.SendCommand)

        rospy.on_shutdown(self.SendLand)

    def ReceiveNavdata(self,navdata):
        self.navdata = navdata

    def SendTakeoff(self):
        # Send a takeoff message to the ardrone driver. Note we only
        # send a takeoff message if the drone is landed - an unexpected
        # takeoff is not good!
        if self.navdata.state == DroneStatus["Landed"]:
                self.pubTakeoff.publish(Empty())

    def SendLand(self):
        # Send a landing message to the ardrone driver
        # Note we send this in all states, landing can do no harm
        self.pubLand.publish(Empty())

    def SendEmergency(self):
        self.pubReset.publish(Empty())

    def SetCommand(self,roll=0,pitch=0,yaw_velocity=0,z_velocity=0):
        self.command.linear.x  = pitch
        self.command.linear.y  = roll
        self.command.linear.z  = z_velocity
        self.command.angular.z = yaw_velocity

    def SendCommand(self):
        if self.navdata.state in itemgetter("Flying","GotoHover","Hovering")(DroneStatus):
                self.pubCommand.publish(self.command)
