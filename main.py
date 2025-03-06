from drone_controller import DroneController

if __name__ == "__main__":
    print("start")
    controller = DroneController(use_laptop_camera=True)
    controller.run()
    print("end")
