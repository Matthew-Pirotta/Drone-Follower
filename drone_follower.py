from typing import Literal
from djitellopy import Tello

AREA_TOLERANCE = .1
CENTER_X_TOLERANCE = .1 # The user should be within 10% of centre of the screen 
FACE_ROT_TOLERANCE = .02
HEIGHT_TOLERANCE = .2


IMAGE_RESOLUTION = (640,480) #TELLO DRONE: (1280,780)? My Laptop: (640,480)
IMAGE_CENTRE = IMAGE_RESOLUTION[0]//2, IMAGE_RESOLUTION[1]//2 
IMAGE_AREA = IMAGE_RESOLUTION[0] * IMAGE_RESOLUTION[1]

TARGET_AREA_PERCENTAGE = .3 #User should take up, 30% of the screen.
TARGET_Y = IMAGE_RESOLUTION[1] * 0.25  # Target head position, 25% from top

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

        #Person was not dectected
        if b_area == 0:
            return fb,lr

        # Move closer or further
        b_area_perc = b_area/IMAGE_AREA
        area_status = self._is_outside_tolerance(b_area_perc, TARGET_AREA_PERCENTAGE, AREA_TOLERANCE)
        tolerance_delta = TARGET_AREA_PERCENTAGE * AREA_TOLERANCE
        print(f"b_area_perc: {b_area_perc:3f}, lowerBound: {(TARGET_AREA_PERCENTAGE - tolerance_delta):3f}, UpperBound: {(TARGET_AREA_PERCENTAGE + tolerance_delta):3f}")

        if area_status == "lower":
           # print("Move closer")
            fb = MOVE_MAG
        elif area_status == "higher":
            #print("Move further away")
            fb = -MOVE_MAG
        elif area_status == "within":
            #print("Within FB Range ")
            pass


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

        #Person was detected but face features were not
        if ecx == -1 or nose_tip_x == -1:
            return yaw

        face_status = self._is_outside_tolerance(nose_tip_x, ecx, FACE_ROT_TOLERANCE)
        tolerance_delta = ecx * FACE_ROT_TOLERANCE
        print(f"nose_tip_x: {nose_tip_x}, lowerBound: {ecx - tolerance_delta}, UpperBound: {ecx + tolerance_delta} ")
        
        if face_status == "lower":
            #Rotate Left, Person turned right
            yaw = ROT_MAG
        elif face_status == "higher":
            #Rotate Right, Person turned left
            yaw = -ROT_MAG
        elif face_status == "within":
           # print("Rotation Centered")
           pass
                    
        return yaw

    def follow_person_height(self, y1: int) -> int:
        """
        Adjusts the drone's height to keep the camera slightly above the person's head.

        Args:
            y1: The y-coordinate of the top of the detected person's bounding box.

        Returns:
            ud: up/down velocity
        """
        ud = 0

        if y1 == -1:
            return ud  # No person detected, no height adjustment
        
        y_status = self._is_outside_tolerance(y1, TARGET_Y, HEIGHT_TOLERANCE)
        tolerance_delta = TARGET_Y * AREA_TOLERANCE
        print(f"y1: {y1}, lowerBound: {TARGET_Y - tolerance_delta}, UpperBound: {TARGET_Y + tolerance_delta}")

        if y_status == "lower":
            print("Move down")
            ud = -MOVE_MAG  # Person is too high, move down
        elif y_status == "higher":
            print("Move up")
            ud = MOVE_MAG  # Person is too low, move up
        else:
            print("Height is within range")

        return ud


    def person_follower_controller(self, tello:Tello, cx:int, bArea, ecx:int, nose_tip_x:int, head_y:int):
        fb = 0
        lr = 0
        yaw = 0
        ud = 0
        
        fb, lr = self.follow_person(cx, bArea)
        yaw = self.match_face_orientation(ecx, nose_tip_x)
        ud = self.follow_person_height(head_y)  # Adjust height based on head position

        #print(f"lr:{lr} fb:{fb} 0, yaw:{yaw}")

        if tello:
            #NOTE temp removed the yaw var
            tello.send_rc_control(lr, fb, ud, yaw)