import cv2
import time

from djitellopy import Tello
from test import PersonRecognition
from drone_follower import DroneFollower
from hand_recognition import HandRecognition

class DroneController:
    """
    Drone Controller loop:
    1. Takeoff
    2. Will hover waiting for "follow mode" signal
    3. "follow mode" untill exit signal 
    """
    def __init__(self,  use_laptop_camera=False):
        self.person_recognition = PersonRecognition()  
        self.drone_follower = DroneFollower()    
        self.hand_recognition = HandRecognition()      
        self.use_laptop_camera = use_laptop_camera
        self.follow_mode = False
        self.kill_switch = False

        self.cap = cv2.VideoCapture(0)  # Using laptop camera
        self.tello = Tello()
        if self.use_laptop_camera:
            pass
        else:
            self.cap = None
            self.tello.connect()
            self.tello.streamon()

            time.sleep(2)  # Wait for the stream to stabilize

            print(f"Battery: {self.tello.get_battery()}%")

    def run(self):
        try:
            if not self.use_laptop_camera:
                self.custom_takeoff()

            while True:
                frame = self.get_frame()
                if frame is None: continue

                hand_frame = self.process_hand_gesture(frame)
                person_frame = self.process_person_tracking(frame)
                overlay_frame = self.overlayImages(frame, hand_frame,person_frame)


                cv2.imshow("Drone Tracking", overlay_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): #Safty kill switch
                    break
                if self.kill_switch:
                    break
        finally:
            # Cleanup
            if not self.use_laptop_camera:
                self.tello.streamoff()
                self.tello.end()
            cv2.destroyAllWindows()

    def custom_takeoff(self):
        self.tello.takeoff()

        # 175 is average height,
        # but tello height reading seems to be innacurate and reads this height as 100
        while self.tello.get_height() < 120: 
            print(f"current height: {self.tello.get_height()}")
            self.tello.send_rc_control(0, 0, 20, 0)  # Move up at speed 20 cm/s
            time.sleep(0.5)

        self.tello.send_rc_control(0, 0, 0, 0)  # Stop moving up
        print(f"Reached {self.tello.get_height()} cm")
    
    def get_frame(self):
        """Captures a frame from the Tello's camera stream, OR from laptop camera."""

        if self.use_laptop_camera:
            ret, frame = self.cap.read()
            if not ret:
                return None
        else:
            frame = self.tello.get_frame_read().frame
            if frame is None or frame.size == 0:
                return None
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        frame = cv2.resize(frame, (640, 480))
        return frame

    def process_hand_gesture(self, frame):
        hand_frame = frame.copy()

        #Handle hand gestures
        hand_gesture = self.hand_recognition.process_frame(hand_frame)

        match hand_gesture:
            case 'FOLLOW':
                self.follow_mode = True
            case 'STOP':
                self.kill_switch = True
            case 'NONE':
                pass
        
        return hand_frame
    
    def process_person_tracking(self,frame):
        person_frame = frame.copy()
        cx, current_area, ecx, nose_tip_x, person_frame = self.person_recognition.detect(person_frame)

        if self.follow_mode:
            self.drone_follower.person_follower_controller(self.tello, cx, current_area, ecx, nose_tip_x)
        else:
            self.tello.send_rc_control(0, 0, 0, 0)  # Hover

        return person_frame

    def overlayImages(self, frame, hand_frame, person_frame):
        #Merge hand tracking and person tracking overlays

        overlay = cv2.addWeighted(frame, 0.1, hand_frame, 0.9, 0)
        overlay = cv2.addWeighted(overlay, 0.5, person_frame, 0.5, 0)
        return overlay

    