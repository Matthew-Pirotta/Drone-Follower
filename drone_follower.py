from typing import Literal
from djitellopy import Tello

AREA_TOLERANCE = .1
CENTER_X_TOLERANCE = .1 # The user should be within 10% of centre of the screen 
FACE_ROT_TOLERANCE = .04

IMAGE_RESOLUTION = (640,480) #TELLO DRONE: (1280,780)? My Laptop: (640,480)
IMAGE_CENTRE = IMAGE_RESOLUTION[0]//2, IMAGE_RESOLUTION[1]//2 
IMAGE_AREA = IMAGE_RESOLUTION[0] * IMAGE_RESOLUTION[1]
TARGET_AREA_PERCENTAGE = .3 #User should take up, 30% of the screen.

MOVE_MAG = 10 # Base movement magnitude at which the drone will move
ROT_MAG = 10 # Base rotation magnitude at which the drone will move

class DroneFollower:

    def __init__(self):
        pass

    def _is_outside_tolerance(self, value: float, target: float, tolerance: float) -> Literal["lower", "higher", "within"]:
        """
        Checks if a value is outside a given tolerance range around a target.

        Args:
            value: The value to check.
            target: The central reference value.
            tolerance: The tolerance percentage

        Returns:
            Literal["lower", "higher", "within"]: Whether the value is below, above, or within tolerance.
        """

        tolerance_delta = target * tolerance
        if value < target - tolerance_delta:
            return "lower"
        elif value > target + tolerance_delta:
            return "higher"
        return "within"


    def follow_person(self, cx:int, b_area:int) -> tuple:
        """
        Args:
            cx: Centroid coordinate-x.
            bAarea: Person bounding box area

        Returns:
            fb: forward/backward velocity
            lr: left/right velocity
        """
        fb = 0
        lr = 0

        # Move closer or further
        b_area_perc = b_area/IMAGE_AREA
        area_status = self._is_outside_tolerance(b_area_perc, TARGET_AREA_PERCENTAGE, AREA_TOLERANCE)
        tolerance_delta = TARGET_AREA_PERCENTAGE * AREA_TOLERANCE
        print(f"b_area_perc: {b_area_perc:3f}, lowerBound: {(TARGET_AREA_PERCENTAGE - tolerance_delta):3f}, UpperBound: {(TARGET_AREA_PERCENTAGE + tolerance_delta):3f}")

        if area_status == "lower":
            print("Move closer")
            fb = MOVE_MAG
        elif area_status == "higher":
            print("Move further away")
            fb = -MOVE_MAG
        elif area_status == "within":
            print("Within FB Range ")


        # Move left or right
        x_status = self._is_outside_tolerance(cx, IMAGE_CENTRE[0], CENTER_X_TOLERANCE)
        tolerance_delta = IMAGE_CENTRE[0]* CENTER_X_TOLERANCE
        print(f"cx: {cx}, lowerBound: {IMAGE_CENTRE[0] - tolerance_delta}, UpperBound: {IMAGE_CENTRE[0] + tolerance_delta} ")
        if x_status == "lower":
            print("Move left")
            lr = -MOVE_MAG
        elif x_status == "higher":
            print("Move right")
            lr = MOVE_MAG
        
        return fb, lr


    def match_face_orientation(self, ecx:int, nose_tip_x:int) -> int:
        """
        Args:
            ecx: eye centroid x-coordinate.
            nose_tip_x: nose tip x-coordinate
            
            Returns:
                yaw: yaw rotation velocity
        """
        yaw = 0

        face_status = self._is_outside_tolerance(nose_tip_x, ecx, FACE_ROT_TOLERANCE)
        tolerance_delta = ecx * FACE_ROT_TOLERANCE
        print(f"nose_tip_x: {nose_tip_x}, lowerBound: {ecx - tolerance_delta}, UpperBound: {ecx + tolerance_delta} ")
        
        if face_status == "lower":
            #Rotate Left, Person turned right
            yaw = -ROT_MAG
        elif face_status == "higher":
            #Rotate Right, Person turned left
            yaw = ROT_MAG
        elif face_status == "within":
            print("Rotation Centered")
                    
        return yaw

    def person_follower_controller(self, tello:Tello, cx:int, bArea, ecx:int, nose_tip_x:int):
        fb = 0
        lr = 0
        yaw = 0

        #Person was not dectected
        if bArea == 0: return
        
        #Person was detected but features were not
        if ecx == -1 or nose_tip_x == -1: return
        
        fb, lr = self.follow_person(cx, bArea)
        yaw = self.match_face_orientation(ecx, nose_tip_x)

        print(f"lr:{lr} fb:{fb} 0, yaw:{yaw}")

        if tello:
            #NOTE temp removed the yaw var
            tello.send_rc_control(lr, fb, 0, 0)