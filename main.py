from drone_controller import DroneController

if __name__ == "__main__":
    print("start")
    #Enable follow mode on start up, make debugging easier
    #since hand in not always recognised
    controller = DroneController(use_laptop_camera=False, follow_mode=True)
    controller.run()
    print("end")
