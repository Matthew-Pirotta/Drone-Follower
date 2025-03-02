import cv2
import time

from djitellopy import Tello
from person_recognition import PersonRecognition
from drone_follower import DroneFollower

class DroneController:
    def __init__(self,  use_laptop_camera=False):
        self.person_recognition = PersonRecognition()  
        self.drone_follower = DroneFollower()          
        self.use_laptop_camera = use_laptop_camera


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

                cx, current_area, ecx, nose_tip_x, processed_img = self.person_recognition.process_frame(frame)
                self.drone_follower.person_follower_controller(self.tello, cx, current_area, ecx, nose_tip_x)

                # Quit when 'q' is pressed
                # Display processed frame

                #TODO displaying processed_img doesnt work
                cv2.imshow("Drone Tracking", processed_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
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
    