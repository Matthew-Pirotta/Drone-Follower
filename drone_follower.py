from pd_controller import PDController
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
        # PD controllers for movement and rotation (no Ki term)
        self.fb_pd = PDController(Kp=0.2, Kd=0.1, setpoint=TARGET_AREA_PERCENTAGE)
        self.lr_pd = PDController(Kp=0.2, Kd=0.2, setpoint=IMAGE_CENTRE[0])
        self.yaw_pd = PDController(Kp=0.1, Kd=0.1, setpoint=0)

    def follow_person(self, cx: int, b_area: int) -> tuple:
        """
        Args:
            cx: Centroid x-coordinate.
            b_area: Bounding box area of the person.

        Returns:
            fb: forward/backward velocity
            lr: left/right velocity
        """
        if b_area == 0:
            return 0,0

        # Normalize bounding box area
        b_area_perc = b_area / IMAGE_AREA

        # Compute PD outputs
        fb = self.fb_pd.compute(b_area_perc)  # Forward/Backward movement
        lr = -self.lr_pd.compute(cx)  # Left/Right movement

        return fb, lr

    def match_face_orientation(self, ecx: int, nose_tip_x: int) -> int:
        """
        Args:
            ecx: eye centroid x-coordinate.
            nose_tip_x: nose tip x-coordinate.

        Returns:
            yaw: yaw rotation velocity
        """
        if ecx == -1 or nose_tip_x == -1:
            return 0

        error = nose_tip_x - ecx  # Difference between nose and eye center
        yaw = self.yaw_pd.compute(error)  # Compute PD yaw adjustment

        return yaw

    def person_follower_controller(self, tello: Tello, cx: int, bArea: int, ecx: int, nose_tip_x: int):
        fb, lr, yaw = 0, 0, 0

        print(f"bArea: {(bArea / IMAGE_AREA):.2f} target bArea:{TARGET_AREA_PERCENTAGE} \tcx:{cx}  target cx: {0}")
        fb, lr = self.follow_person(cx, bArea)
        yaw = self.match_face_orientation(ecx, nose_tip_x)

        print(f"lr:{lr} fb:{fb} yaw:{yaw}")

        if tello:
            tello.send_rc_control(lr, fb, 0, yaw)  # Send PD-controlled movements
