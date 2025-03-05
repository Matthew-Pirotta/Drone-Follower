from djitellopy import Tello
from test import PersonRecognition
from drone_follower import DroneFollower
from drone_controller import DroneController

if __name__ == "__main__":
    print("start")
    controller = DroneController(use_laptop_camera=True)
    controller.run()
    print("end")
