MAX_SPEED = 0.2

KeyMapping=dict(PitchForward = 'w'
                ,PitchBackward = 's'
                ,RollLeft = 'a'
                ,RollRight = 'd'
                ,YawLeft = 'q'
                ,YawRight = 'e'
                ,IncreaseAltitude = '+'
                ,DecreaseAltitude = '-'
                ,Takeoff = 't'
                ,Land = 'l'
                ,Emergency = 'k')
KeyMapping = dict(zip(KeyMapping.keys(),map(ord,KeyMapping.values())))


class KeyboardController(object):
    def __init__(self,controller):
        self.controller=controller
        self.pitch = 0
        self.roll = 0
        self.yaw_velocity = 0
        self.z_velocity = 0

    def keyPressEvent(self, key):
        # Handle the important cases first!
        if key == KeyMapping['Emergency']:
            self.controller.SendEmergency()
        elif key == KeyMapping['Takeoff']:
            self.controller.SendTakeoff()
        elif key == KeyMapping['Land']:
            self.controller.SendLand()
        else:
            if key == KeyMapping['YawLeft']:
                self.yaw_velocity = min(MAX_SPEED,self.yaw_velocity+MAX_SPEED)                    
            elif key == KeyMapping['YawRight']:
                self.yaw_velocity = max(-MAX_SPEED,self.yaw_velocity-MAX_SPEED)

            elif key == KeyMapping['PitchForward']:
                self.pitch = min(MAX_SPEED,self.pitch+MAX_SPEED)
            elif key == KeyMapping['PitchBackward']:
                self.pitch = max(-MAX_SPEED,self.pitch-MAX_SPEED)

            elif key == KeyMapping['RollLeft']:
                self.roll = min(MAX_SPEED,self.roll+MAX_SPEED)
            elif key == KeyMapping['RollRight']:
                self.roll = max(-MAX_SPEED,self.roll-MAX_SPEED)

            elif key == KeyMapping['IncreaseAltitude']:
                self.z_velocity = min(MAX_SPEED,self.z_velocity+MAX_SPEED)
            elif key == KeyMapping['DecreaseAltitude']:
                self.z_velocity = max(-MAX_SPEED,self.z_velocity-MAX_SPEED)

            else:
                self.pitch = 0
                self.roll = 0
                self.yaw_velocity = 0
                self.z_velocity = 0

            # finally we set the command to be sent. The controller handles sending this at regular intervals
            self.controller.SetCommand(self.roll, self.pitch, self.yaw_velocity, self.z_velocity)
