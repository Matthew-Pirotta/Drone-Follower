from djitellopy import Tello
import cv2

# Connect to Tello
tello = Tello()
tello.connect()

# Start video streaming
tello.streamon()

try:
    while True:
         # Get video frame
        frame = tello.get_frame_read().frame

        # Convert from RGB to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Resize (optional)
        frame = cv2.resize(frame, (640, 480))

        # Show frame
        cv2.imshow("Tello Camera", frame)

        # Break loop on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Cleanup
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()
