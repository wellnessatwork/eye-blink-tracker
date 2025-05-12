import cv2
import numpy as np
import mediapipe as mp
import time
import sys
import math
import json

class EyeBlinkCounter:
    def __init__(self, ear_threshold=0.2, min_frames_closed=1, max_frames_closed=5):
        self.mp_face_mesh = mp.solutions.face_mesh
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
        except Exception as e:
            print(f"Error initializing MediaPipe FaceMesh: {e}", file=sys.stderr)
            self.face_mesh = None

        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
        
        self.EAR_THRESHOLD = ear_threshold
        self.MIN_FRAMES_CLOSED = min_frames_closed
        self.MAX_FRAMES_CLOSED = max_frames_closed
        
        self.total_blinks = 0
        self.previous_eye_state = 'open'
        self.state_counter = 0

    def _calculate_ear(self, eye_points):
        if eye_points is None or eye_points.shape[0] != 6:
            return float('inf') 

        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        ear = (v1 + v2) / (2.0 * h + 1e-6)
        return ear

    def process_frame(self, frame_bgr):
        if not self.face_mesh:
            return self.total_blinks, None

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = frame_bgr.shape[:2]
        
        results = self.face_mesh.process(frame_rgb)
        
        current_ear_avg = float('inf') 
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_eye_points = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in self.LEFT_EYE_INDICES])
            right_eye_points = np.array([[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in self.RIGHT_EYE_INDICES])

            left_ear = self._calculate_ear(left_eye_points)
            right_ear = self._calculate_ear(right_eye_points)
            
            valid_ears = [e for e in [left_ear, right_ear] if e != float('inf')]
            if valid_ears:
                current_ear_avg = sum(valid_ears) / len(valid_ears)
        
        current_eye_state = 'open'
        if face_detected and current_ear_avg < self.EAR_THRESHOLD:
            current_eye_state = 'closed'
        
        if current_eye_state == 'closed' and self.previous_eye_state == 'open':
            self.state_counter = 1
        elif current_eye_state == 'closed' and self.previous_eye_state == 'closed':
            self.state_counter += 1
        elif current_eye_state == 'open' and self.previous_eye_state == 'closed':
            if self.MIN_FRAMES_CLOSED <= self.state_counter <= self.MAX_FRAMES_CLOSED:
                self.total_blinks += 1
            self.state_counter = 0 

        self.previous_eye_state = current_eye_state
        return self.total_blinks, current_ear_avg

    def close(self):
        if self.face_mesh:
            self.face_mesh.close()

def run_blink_detection_loop():
    try:
        counter = EyeBlinkCounter()
    except Exception as e:
        print(f"Failed to initialize EyeBlinkCounter: {e}", file=sys.stderr)
        sys.exit(1)

    if not counter.face_mesh:
        print("Exiting due to FaceMesh initialization failure.", file=sys.stderr)
        sys.exit(1)
        
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.", file=sys.stderr)
        counter.close()
        sys.exit(1)

    last_printed_blink_count = -1
    last_printed_ear = None
    print("Eye blink counter started. Outputting JSON...", file=sys.stderr)
    print(json.dumps({"blinks": 0, "ear": "inf"}), flush=True)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            current_blinks, current_ear = counter.process_frame(frame)
            if current_ear is not None and current_ear != float('inf'):
                floored_ear = math.floor(current_ear * 100) / 100
                rounded_ear = float(f"{floored_ear:.2f}")
            else:
                rounded_ear = "inf"

            if (current_blinks != last_printed_blink_count) or (rounded_ear != last_printed_ear):
                print(json.dumps({"blinks": current_blinks, "ear": rounded_ear}), flush=True)
                last_printed_blink_count = current_blinks
                last_printed_ear = rounded_ear

    except KeyboardInterrupt:
        print("\nExiting due to user interruption...", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred during processing: {e}", file=sys.stderr)
    finally:
        print("Releasing resources...", file=sys.stderr)
        cap.release()
        counter.close()
        print("Eye blink counter stopped.", file=sys.stderr)

if __name__ == "__main__":
    run_blink_detection_loop() 