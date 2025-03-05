import cv2
import logging
import math
import numpy as np
import face_recognition
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Suppress YOLOv8 logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)

class PersonRecognition:
    def __init__(self):
        # Load YOLOv8 model
        #NOTE TODO use YOLOv8n instead? to reduce launch latancy
        self.model = YOLO("yolov8n.pt", verbose=False)

        # Initialize DeepSORT Tracker
        self.tracker = DeepSort(
            max_age=10,  
            n_init=3,    
            max_iou_distance=0.5,
            max_cosine_distance=0.4,  
            embedder_model_name="mobilenetv2",
            half=True,
            embedder_gpu=True
        )

        # Persistent ID tracking
        self.face_db = {}  
        self.person_id_map = {}  
        self.used_person_ids = set()  
        self.next_person_id = 1

        self.last_position = None
        self.last_area = None
        self.pixel_threshold = 100  

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

    def process_frame(self, frame):
        """
        Processes a frame to detect people, track them, and identify facial landmarks.
        Returns:
            cx (int): X coordinate of person center
            current_area (int): Bounding box area of the person
            ecx (int): X coordinate of eye center
            nose_tip_x (int): X coordinate of nose tip
        """
        sharpened = cv2.filter2D(frame, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        results = self.model(sharpened)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])  
                conf = box.conf[0].item()  

                if cls == 0 and conf > 0.8:  
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  
                    width, height = x2 - x1, y2 - y1
                    person_crop = sharpened[y1:y2, x1:x2]

                    if person_crop.size == 0:
                        continue
                    
                    detections.append(([x1, y1, width, height], conf, 0, np.zeros((128,))))  

        tracked_objects = self.tracker.update_tracks(detections, frame=sharpened)

        person_x, person_y, current_area = None, None, 0

        for track in tracked_objects:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)

            assigned_id = self.person_id_map.get(track_id)
            if assigned_id is None:
                while self.next_person_id in self.used_person_ids:
                    self.next_person_id += 1
                assigned_id = self.next_person_id
                self.used_person_ids.add(self.next_person_id)
                self.next_person_id += 1

            self.person_id_map[track_id] = assigned_id

            if assigned_id == 1:
                person_x, person_y, current_area = cx, cy, area
                self.last_position = (person_x, person_y)
                self.last_area = current_area

                # Detect head rotation
                ecx, nose_tip_x = self.detect_head_rotation(sharpened)

                # Draw bounding box and tracking ID
                cv2.rectangle(sharpened, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(sharpened, f"ID: {assigned_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                areprc = area/307200
                areprc = round(areprc,2)
                cv2.putText(sharpened, f"Area perc:{areprc}",(x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) #TODO value is hard coded big no no
                cv2.circle(sharpened, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

                return cx, current_area, ecx, nose_tip_x, sharpened

        return 0, 0, -1, -1, sharpened  # No person detected

