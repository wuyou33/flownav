from drone_control import DroneStatus
MAX_SPEED = 0.3
INC = 0.1

KeyMapping=dict(FlightToggle = ' '
                ,StartStopToggle = '\n'
                ,IncreaseVelocity = 'j'
                ,DecreaseVelocity = 'k'
                ,IncreaseAltitude = 'h'
                ,DecreaseAltitude = 'l'
                ,Emergency = 'e'
                ,EvadeLeft = '<'
                ,EvadeRight= '>'
                ,TurnLeft = ','
                ,TurnRight= '.')
                
CharMap = KeyMapping
KeyMapping = dict(zip(KeyMapping.keys(),map(ord,KeyMapping.values())))

class AutoController(object):
    def __init__(self,controller):
        self.__controller=controller

        self.pitch = 0
        self.roll = 0
        self.yaw_velocity = 0
        self.z_velocity = 0

        self.__last_state = dict(pitch=0,roll=0,yaw_velocity=0,z_velocity=0)

    def SendCommands(self,n=1):
        for i in range(n):
            self.__controller.SendCommand(self.roll, self.pitch, self.yaw_velocity, self.z_velocity)

    def EvadeLeft(self):
        self.roll = -INC
        # self.pitch = INC
        print "Evade Left"
        self.SendCommands(n=8)
        self.roll = 0
        self.SendCommands()

    def EvadeRight(self):
        self.roll = INC
        # self.pitch = INC
        print "Evade Right"
        self.SendCommands(n=8)
        self.roll = 0
        self.SendCommands()

    def Pause(self):
        for attr in self.__last_state:
            self.__last_state[attr] = getattr(self,attr)

        self.pitch = 0
        self.roll = 0
        self.yaw_velocity = 0
        self.z_velocity = 0

        self.SendCommands()        

    def Play(self):
        for attr,val in self.__last_state.items():
            setattr(self,attr,val)
        if self.pitch == 0: self.pitch = INC
        self.SendCommands()

    def keyPressEvent(self, key):
        if key == KeyMapping['Emergency']:
            self.__controller.SendEmergency()

        elif key == KeyMapping['FlightToggle']:
            if self.__controller.navdata.state == DroneStatus["Landed"]:
                self.__controller.SendTakeoff()
            else:
                self.__controller.SendLand()

        elif key == KeyMapping['StartStopToggle']:
            if self.__controller.navdata.state in itemgetter("Flying","GotoHover","Hovering")(DroneStatus):
                self.Play()
            else:
                self.Pause()

        elif key == KeyMapping['EvadeLeft']:
            self.EvadeLeft()
        elif key == KeyMapping['EvadeRight']:
            self.EvadeRight()

        elif key == KeyMapping['TurnLeft']:
                self.yaw_velocity = -2*INC
                self.SendCommands(8)
                self.yaw_velocity = 0
                self.SendCommands()
        elif key == KeyMapping['TurnRight']:
                self.yaw_velocity = 2*INC
                self.SendCommands(8)
                self.yaw_velocity = 0
                self.SendCommands()

        else:
            if key == KeyMapping['IncreaseVelocity']:
                self.pitch = min(MAX_SPEED,self.pitch+INC)
            elif key == KeyMapping['DecreaseVelocity']:
                self.pitch = max(-MAX_SPEED,self.pitch-INC)

            elif key == KeyMapping['IncreaseAltitude']:
                self.z_velocity = min(MAX_SPEED,self.z_velocity+INC)
            elif key == KeyMapping['DecreaseAltitude']:
                self.z_velocity = max(-MAX_SPEED,self.z_velocity-INC)

            self.SendCommands()
