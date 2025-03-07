from drone_controller import DroneController

if __name__ == "__main__":
    print("start")
    controller = DroneController(use_laptop_camera=False)
    controller.run()
    print("end")
