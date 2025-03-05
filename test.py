# Import all the necessary libraries
import cv2
import numpy as np
import face_recognition
import logging
# import math
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class PersonRecognition:
    def __init__(self):
        # Load YOLOv8 model
        self.model = self.initialize_yolo()

        # Initialize DeepSORT Tracker
        self.tracker = self.initialize_tracker()

        # Persistent ID tracking
        # face_db -> Stores face encodings used for Persistent ID tracking {unique_id: face_encoding}
        # person_id_map -> Stores the mapping of the tracked ID to the assigned ID {track_id: assigned_id}
        # used_person_ids -> Stores the set of used IDs {unique_id}
        # next_person_id -> Stores the next available ID {unique_id}
        # last_position -> Stores the last position of the person tracked {person_id: (x, y)}
        # last_area -> Stores the last area of the person tracked {person_id: area}
        # pixel_threshold -> Stores the threshold for movement in pixels {threshold}
        self.face_db = {}  
        self.person_id_map = {}  
        self.used_person_ids = set()  
        self.next_person_id = 1

        self.last_position = None
        self.last_area = None
        self.pixel_threshold = 100  

    def initialize_yolo(self):
        # Suppress YOLOv8 logging
        logging.getLogger("ultralytics").setLevel(logging.WARNING)
        # Load YOLOv8 model
        return YOLO("yolov8n.pt", verbose=False)

    def initialize_tracker(self):
        return DeepSort(
            # The ammount of time an ID survives in memory without being removed
            # If this is lost for 10 consecutive frames than it is lost - this won't be a problem since we are relying on face recognition
            max_age=10,
            # The ammount of times the processer checks if it is the same ID
            n_init=3,
            # To associate detections - maximum Intersection over Union
            # Lower values make tracking stricter, requiring detections to be very close to previous locations.
            max_iou_distance=0.5,
            # The confidence it has to say that they are the same person
            # The maximum cosine distance between feature embeddings to consider two detections as the same object.
            max_cosine_distance=0.4,
            # Specifies the model used for feature extraction
            embedder_model_name="mobilenetv2",
            # Enables half-precision floating point (FP16) computations.
            half=True,
            # Turn on the GPU acceleration for feature extract
            embedder_gpu=True
        )

    def preprocess_frame(self, frame):
        # Sharpen image
        # Kernel to sharpen the image
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # Use the kernel to filter the image
        return cv2.filter2D(frame, -1, kernel)

    def detect_people(self, model, frame):
        # Run YOLOv8 inference with the sharpened image
        results = model(frame)
        # List to hold detections
        detections = []
        # For each detection in the results
        for r in results:
            # For each bounding box in the detection
            for box in r.boxes:
                # Get class ID
                cls = int(box.cls[0])
                # Get the confidence score
                conf = box.conf[0].item()
                # Class 0 = "person", confidence > 80%
                if cls == 0 and conf > 0.8:
                    # Bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    # Calculate the width and the height
                    width, height = x2 - x1, y2 - y1
                    # If the width and the height is less than zero we continue 
                    # Missing lines here
                    if width > 0 and height > 0:
                        # Default 128-dim zero vector
                        detections.append(([x1, y1, width, height], conf, 0, np.zeros((128,))))
        return detections

    def track_people(self, tracker, detections, frame):
        # Update DeepSORT tracker - so the tracker will only track the detected people
        return tracker.update_tracks(detections, frame=frame)


    def assign_person_id(self, face_encoding, face_db, used_person_ids, next_person_id):
        # Check if this face matches a known person
        for person_id, saved_encoding in face_db.items():
            # Compare face encodings
            if face_recognition.compare_faces([saved_encoding], face_encoding, tolerance=0.55)[0]:
                # If there was a match return it
                return person_id, next_person_id
        # If there was no match  
        # While the next person is in used ids
        while next_person_id in used_person_ids:
            # Increment to find an unused ID
            next_person_id += 1
        # Store face encoding in a dictionary
        face_db[next_person_id] = face_encoding
        # Add this to the set of used IDs
        used_person_ids.add(next_person_id)
        # Return the assigned ID and the next available ID
        return next_person_id, next_person_id + 1

    def process_tracked_objects(self, tracked_objects, frame, face_db, person_id_map, used_person_ids, next_person_id, last_position, last_area, pixel_threshold):
        # Track only ID=1 -> variables to hold everything with id=1
        person_x, person_y, current_area = None, None, 0
        # For each tracked object
        for track in tracked_objects:
            # If there is no tracking continue
            if not track.is_confirmed():
                continue
            # If there was a tracking
            # Get track ID
            track_id = track.track_id
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            # Get the centre
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            # Get the area of the bounding box
            area = (x2 - x1) * (y2 - y1)
            # Extract face from person crop
            person_crop = frame[y1:y2, x1:x2]
            # Get face encodings -> storing the locations
            face_locations = face_recognition.face_locations(person_crop, model="hog")
            # Final ID assignment
            assigned_id = None
            # If there was face recognitions
            if face_locations:
                # Get adjusted bounding box coordinates for face recognition
                adjusted_faces = [(y1 + top, x1 + right, y1 + bottom, x1 + left) for (top, right, bottom, left) in face_locations]
                # Get face encodings
                face_encodings = face_recognition.face_encodings(frame, known_face_locations=adjusted_faces)
                # If a face was detected and recognized
                if face_encodings:
                    # Store mapping - saved the ID that we will be using
                    assigned_id, next_person_id = self.assign_person_id(face_encodings[0], face_db, used_person_ids, next_person_id)
            # Store mapping
            person_id_map[track_id] = assigned_id
            # Only track and draw for ID = 1
            if assigned_id == 1:
                # Get the x and y  co-ordinates, and the area of the person
                person_x, person_y, current_area = cx, cy, area
                # Draw tracking box only for ID = 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {assigned_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
                ecx, nose_tip_x = self.detect_head_rotation(frame)
                # Track the movement -> this can be then changed with Matthew's code
                # last_position, last_area = self.track_movement(assigned_id, person_x, person_y, current_area, last_position, last_area, pixel_threshold)
                return cx, current_area, ecx, nose_tip_x
        return 0, 0, -1, -1

    # def track_movement(self, assigned_id, person_x, person_y, current_area, last_position, last_area, pixel_threshold):
    #     # If a person had been found before - in this case with ID = 1, the first person ever seen 
    #     if last_position is not None and last_area is not None and current_area > 0:
    #         # Calculate distance moved in pixels
    #         dx, dy = person_x - last_position[0], person_y - last_position[1]
    #         # Use the Eulidean distance
    #         distance_moved = math.sqrt(dx**2 + dy**2)
    #         # If the distance moved is greater than the pixel threshold
    #         if distance_moved > pixel_threshold:
    #             # Print the movement details, the new area, the ID to ensure that it has ID=1 and the center
    #             print(f'ID={assigned_id} -> Area: {current_area}, Center: ({person_x}, {person_y})')
    #             print(f"Person moved {distance_moved:.2f} pixels, updating movement.")
    #             # Return the updated position and the area
    #             return (person_x, person_y), current_area
    #     # If a person had never been found before
    #     else:
    #         # Print the new area and new center
    #         print(f'ID={assigned_id} -> Area: {current_area}, Center: ({person_x}, {person_y})')
    #         # Assign the new position and the new area that has been found to be found again when the person moves
    #         return (person_x, person_y), current_area
    #     return last_position, last_area
    
    def detect_head_rotation(self, frame):
        """
        Detects head rotation by finding the eye center and nose tip.
        Returns:
            eye_center_x (int): X coordinate of the eye center
            nose_tip_x (int): X coordinate of the nose tip
        """
        eye_center_x, nose_tip_x = -1, -1
        face_landmarks_list = face_recognition.face_landmarks(frame)

        if not face_landmarks_list:
            return eye_center_x, nose_tip_x  

        landmarks = face_landmarks_list[0]
        left_eye = landmarks.get("left_eye", [(0, 0)])  
        right_eye = landmarks.get("right_eye", [(0, 0)])  
        nose_bridge = landmarks.get("nose_bridge", [(0, 0)])  

        eye_center_x = (left_eye[0][0] + right_eye[-1][0]) // 2
        eye_center_y = (left_eye[0][1] + right_eye[-1][1]) // 2

        nose_tip_x = nose_bridge[-1][0]

        #debugging
        cv2.circle(frame, (eye_center_x, eye_center_y), 5, (255, 0, 0), cv2.FILLED)

        return eye_center_x, nose_tip_x

    def detect(self):
        # Open webcam
        cap = cv2.VideoCapture(0)
        # While the webcam is open
        while cap.isOpened():
            # Read the next frame from the webcam
            success, frame = cap.read()
            if not success:
                break
            # Sharpen the camera 
            sharpened = self.preprocess_frame(frame)
            # Detect faces and bounding boxes from the frame -> this is where we will use the YOLO model to detect faces
            detections = self.detect_people(self.model, sharpened)
            # Update the tracker with the new detections -> this is where we will use the DeepSORT tracker to track the faces
            tracked_objects = self.track_people(self.tracker, detections, sharpened)
            # Process the tracked objects -> this is where we will use the other functions to track the movement of the faces
            cx, current_area, ecx, nose_tip_x = self.process_tracked_objects(
                tracked_objects, sharpened, self.face_db, self.person_id_map, self.used_person_ids, self.next_person_id, self.last_position, self.last_area, self.pixel_threshold
            )
            # Show frame
            cv2.imshow("Tracking", sharpened)
        
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

        return cx, current_area, ecx, nose_tip_x, sharpened

if __name__ == "__main__":
    p = PersonRecognition()
    p.detect()
