from typing import Literal
import cv2
import mediapipe as mp
import math


class HandRecognition:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.35, min_tracking_confidence=0.35)

    def distance(self, point1, point2):
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def is_peace_sign(self, landmarks) -> bool:
        """Check if the hand is making a peace sign."""
        index_finger_up = landmarks[8].y < landmarks[6].y  # Index finger extended
        middle_finger_up = landmarks[12].y < landmarks[10].y  # Middle finger extended
        ring_finger_down = landmarks[16].y > landmarks[14].y  # Ring finger curled
        pinky_finger_down = landmarks[20].y > landmarks[18].y  # Pinky curled

        is_peace = index_finger_up and middle_finger_up and ring_finger_down and pinky_finger_down
        return is_peace

    
    
    def is_rock_n_roll(self, landmarks):
        """Check if the hand is making the Rock 'n' Roll sign (🤘)."""
        index_finger_up = landmarks[8].y < landmarks[6].y
        pinky_finger_up = landmarks[20].y < landmarks[18].y

        middle_finger_down = landmarks[12].y > landmarks[10].y
        ring_finger_down = landmarks[16].y > landmarks[14].y

        return index_finger_up and pinky_finger_up and middle_finger_down and ring_finger_down
    
    def is_ok_sign(self, landmarks):
        """Check if the hand is making an OK sign (👌)."""
        middle_finger_up = landmarks[12].y < landmarks[10].y
        ring_finger_up = landmarks[16].y < landmarks[14].y
        pinky_finger_up = landmarks[20].y < landmarks[18].y

        thumb_index_distance = self.distance(landmarks[4], landmarks[8])  # Thumb touching index fingertip
        index_finger_curled = landmarks[8].y > landmarks[6].y  # Index finger is curled

        return middle_finger_up and ring_finger_up and pinky_finger_up and index_finger_curled and thumb_index_distance < 0.05

    def process_frame(self, frame) -> Literal["FOLLOW", "STOP", "PAUSE","NONE"]:
        """Process a single frame for hand gestures."""
        #Frame is assumed to already be pre-processed
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                if self.is_peace_sign(hand_landmarks.landmark):
                    print("✌️ Peace!")
                    return "FOLLOW"
                elif self.is_ok_sign(hand_landmarks.landmark):
                    print("👌 OK!")
                    return "PAUSE"
                elif self.is_rock_n_roll(hand_landmarks.landmark):
                    print("Roll sign (🤘)")
                    return "STOP"
        return "NONE"
