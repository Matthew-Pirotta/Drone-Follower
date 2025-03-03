from typing import Literal
import cv2
import mediapipe as mp

class HandRecognition:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    def is_peace_sign(self, landmarks) -> bool:
        """Check if the hand is making a peace sign."""
        index_finger_up = landmarks[8].y < landmarks[6].y  # Index finger extended
        middle_finger_up = landmarks[12].y < landmarks[10].y  # Middle finger extended
        ring_finger_down = landmarks[16].y > landmarks[14].y  # Ring finger curled
        pinky_finger_down = landmarks[20].y > landmarks[18].y  # Pinky curled

        is_peace = index_finger_up and middle_finger_up and ring_finger_down and pinky_finger_down
        return is_peace

    def is_thumbs_down(self, landmarks) -> bool:
        """Check if the hand is making a thumbs-down gesture."""
        thumb_down = landmarks[4].y > landmarks[3].y  # Thumb extended downward
        index_finger_down = landmarks[8].y > landmarks[6].y  # Index curled
        middle_finger_down = landmarks[12].y > landmarks[10].y  # Middle curled
        ring_finger_down = landmarks[16].y > landmarks[14].y  # Ring curled
        pinky_finger_down = landmarks[20].y > landmarks[18].y  # Pinky curled

        is_thumbs_d = thumb_down and index_finger_down and middle_finger_down and ring_finger_down and pinky_finger_down 
        return is_thumbs_d

    def process_frame(self, frame) -> Literal["FOLLOW", "STOP", "NONE"]:
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
                elif self.is_thumbs_down(hand_landmarks.landmark):
                    print("👎 Thumbs Down!")
                    return "STOP"
        return "NONE"
