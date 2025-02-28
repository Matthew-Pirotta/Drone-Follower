# Import all the necessary libraries
import cv2
import logging
import math
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import face_recognition

# Suppress YOLOv8 logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# Load YOLOv8 model
model = YOLO("yolov8n.pt", verbose=False)

# Initialize DeepSORT Tracker
tracker = DeepSort(
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

# Persistent ID tracking
# Stores face encodings {unique_id: face_encoding}
face_db = {} 
# Maps track_id → unique_id 
person_id_map = {}  
# Keeps track of assigned person IDs
used_person_ids = set()  
# Start with id 1
next_person_id = 1

last_position = None
last_area = None
# Movement sensitivity
pixel_threshold = 100  

# Open webcam
cap = cv2.VideoCapture(0)

# While the webcam is open
while cap.isOpened():
    # Read the next frame from the webcam
    success, frame = cap.read()
    if not success:
        break

    # Sharpen image
    # Kernel to sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # Use the kernel to filter the image
    sharpened = cv2.filter2D(frame, -1, kernel)
    
    # Run YOLOv8 inference with the sharpened image
    results = model(sharpened)

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
                # Crop the person
                person_crop = sharpened[y1:y2, x1:x2]

                # If the cropped image has a size of 0 than there must have been an error
                if person_crop.size == 0:
                    continue
                
                # Default 128-dim zero vector
                detections.append(([x1, y1, width, height], conf, 0, np.zeros((128,))))  

    # Update DeepSORT tracker - so the tracker will only track the detected people
    tracked_objects = tracker.update_tracks(detections, frame=sharpened)

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
        person_crop = sharpened[y1:y2, x1:x2]
        # Get face encodings -> storing the locations
        face_locations = face_recognition.face_locations(person_crop, model="hog")
        
        # Final ID assignment
        assigned_id = None  

        # If there was face recognitions
        if face_locations:
            # Get adjusted bounding box coordinates for face recognition
            adjusted_faces = [(y1 + top, x1 + right, y1 + bottom, x1 + left) for (top, right, bottom, left) in face_locations]
            # Get face encodings
            face_encodings = face_recognition.face_encodings(sharpened, known_face_locations=adjusted_faces)

            # If a face was detected and recognized
            if face_encodings:
                # Use first detected face
                face_encoding = face_encodings[0]  

                # Check if this face matches a known person
                for person_id, saved_encoding in face_db.items():
                    # Compare face encodings
                    match = face_recognition.compare_faces([saved_encoding], face_encoding, tolerance=0.55)
                    # If match, assign the person ID - take the first match - the one with id = 1
                    if match[0]:  
                        assigned_id = person_id
                        break

                # If no match, assign a new unique ID
                if assigned_id is None:
                    
                    # While the next person is in used ids
                    while next_person_id in used_person_ids:  
                        # Increment to find an unused ID
                        next_person_id += 1


                    # Store mapping - saved the ID that we will be using
                    assigned_id = next_person_id
                    # Store face encoding in a dictionary
                    face_db[next_person_id] = face_encoding
                    # Add this to the set of used IDs
                    used_person_ids.add(next_person_id)
                    # Increment the ID that should be used next
                    next_person_id += 1

        # Store mapping
        person_id_map[track_id] = assigned_id

        # Only track and draw for ID = 1
        if assigned_id == 1:
            # Get the x and y  co-ordinates, and the area of the person
            person_x, person_y, current_area = cx, cy, area

            # Draw tracking box only for ID = 1
            cv2.rectangle(sharpened, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(sharpened, f"ID: {assigned_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(sharpened, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

            # If a person had been found before - in this case with ID = 1, the first person ever seen 
            if last_position is not None and last_area is not None and current_area > 0:
                # Calculate distance moved in pixels
                dx = person_x - last_position[0]
                dy = person_y - last_position[1]
                # Use the Eulidean distance
                distance_moved = math.sqrt(dx**2 + dy**2)

                # If the distance moved is greater than the pixel threshold
                if distance_moved > pixel_threshold:
                    # Print the movement details, the new area, the ID to ensure that it has ID=1 and the center
                    print(f'ID={assigned_id} -> Area: {current_area}, Center: ({person_x}, {person_y})')
                    print(f"Person moved {distance_moved:.2f} pixels, updating movement.")

                    # Update the position and the area
                    last_position = (person_x, person_y)
                    last_area = current_area
            # If a person had never been found before
            else:
                # Print the new area and new center
                print(f'ID={assigned_id} -> Area: {current_area}, Center: ({person_x}, {person_y})')
                # Assign the new position and the new area that has been found to be found again when the person moves
                last_position = (person_x, person_y)
                last_area = current_area
                
    # Show frame
    cv2.imshow("YOLOv8 + DeepSORT + Face Recognition", sharpened)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  

cap.release()
cv2.destroyAllWindows()
