#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Joy

# define the default mapping between joystick buttons and their corresponding actions
ButtonEmergency = 3
ButtonLand = 0
ButtonTakeoff = 1

# define the default mapping between joystick axes and their corresponding directions
AxisRoll = 0
AxisPitch = 1
AxisYaw = 2
AxisZ = 3

# define the default scaling to apply to the axis inputs. useful where an axis is inverted
ScaleRoll = 1.0
ScalePitch = 1.0
ScaleYaw = 1.0
ScaleZ = 1.0


class JoystickController(object):
    def __init__(self,controller):
        self.joy_sub = rospy.Subscriber('/joy', Joy, self.ReceiveJoystickMessage)
        self.controller=controller

    # handles the reception of joystick packets
    def ReceiveJoystickMessage(self,data):
        if data.buttons[ButtonEmergency]==1:
            rospy.loginfo("Emergency Button Pressed")
            self.controller.SendEmergency()
        elif data.buttons[ButtonLand]==1:
            rospy.loginfo("Land Button Pressed")
            self.controller.SendLand()
        elif data.buttons[ButtonTakeoff]==1:
            rospy.loginfo("Takeoff Button Pressed")
            self.controller.SendTakeoff()
        else:
            self.controller.SetCommand(data.axes[AxisRoll]/ScaleRoll,data.axes[AxisPitch]/ScalePitch,data.axes[AxisYaw]/ScaleYaw,data.axes[AxisZ]/ScaleZ)


if __name__=='__main__':
    import sys
    from drone_control import DroneController
    
    # Firstly we setup a ros node, so that we can communicate with the other packages
    rospy.init_node('ardrone_joystick')

    ButtonEmergency = int ( rospy.get_param("~ButtonEmergency",ButtonEmergency) )
    ButtonLand = int ( rospy.get_param("~ButtonLand",ButtonLand) )
    ButtonTakeoff = int ( rospy.get_param("~ButtonTakeoff",ButtonTakeoff) )
    AxisRoll = int ( rospy.get_param("~AxisRoll",AxisRoll) )
    AxisPitch = int ( rospy.get_param("~AxisPitch",AxisPitch) )
    AxisYaw = int ( rospy.get_param("~AxisYaw",AxisYaw) )
    AxisZ = int ( rospy.get_param("~AxisZ",AxisZ) )
    ScaleRoll = float ( rospy.get_param("~ScaleRoll",ScaleRoll) )
    ScalePitch = float ( rospy.get_param("~ScalePitch",ScalePitch) )
    ScaleYaw = float ( rospy.get_param("~ScaleYaw",ScaleYaw) )
    ScaleZ = float ( rospy.get_param("~ScaleZ",ScaleZ) )

    print ButtonEmergency
    print ButtonLand
    print ButtonTakeoff
    print AxisRoll
    print AxisPitch
    print AxisYaw
    print AxisZ
    print ScaleRoll
    print ScalePitch
    print ScaleYaw
    print ScaleZ

    controller = DroneController()
    subJoystick = JoystickController(controller)

    rospy.spin()
