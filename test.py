# Initial things to use the drone - new features
# Tello does not work with out model 
from djitellopy import tello
from time import sleep
import cv2


me = tello.Tello()
# To connect 
me.connect()

# # To start 
me.streamon()

# # To takeoff
me.takeoff()

# Open a window to display the video feed
while True:
    # Get the video frame
    frame = me.get_frame_read().frame

    # Resize the frame for better display (optional)
    frame = cv2.resize(frame, (640, 480))

    # Show the frame
    cv2.imshow("Tello Camera", frame)

    # me.land()

    

# # To control movement
# # me.send_rc_control(left/right, forward/backward, up/down, yaw_velocity)

# # To stop - for 5 seconds
# me.sleep(60)

# # To land
# me.land()