import sys
import cv2
import numpy as np
import mediapipe as mp
import json
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
import time
from threading import Lock

class ResourceMonitor:
    def __init__(self):
        self.fps = 0
        self.frame_times = []
        self.max_frame_times = 30
        self.lock = Lock()

    def update_fps(self):
        current_time = time.time()
        self.frame_times.append(current_time)
        
        while len(self.frame_times) > self.max_frame_times:
            self.frame_times.pop(0)
        
        if len(self.frame_times) > 1:
            self.fps = len(self.frame_times) / (self.frame_times[-1] - self.frame_times[0])
        else:
            self.fps = 0

class BlinkDetector:
    def __init__(self):
        # Initialize MediaPipe for face detection and landmark extraction
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3
        )
        
        # Eye landmark indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Blink detection parameters
        self.EAR_THRESHOLD = 0.2
        self.total_blinks = 0
        self.previous_state = 'open'
        self.state_counter = 0
        self.min_frames_closed = 1
        self.max_frames_closed = 5
        
        # Create resource monitor
        self.resource_monitor = ResourceMonitor()

    def calculate_ear(self, eye_points):
        # Calculate the vertical distances
        v1 = np.linalg.norm(eye_points[1] - eye_points[5])
        v2 = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Calculate the horizontal distance
        h = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Calculate EAR
        ear = (v1 + v2) / (2.0 * h + 1e-6)  # Add small epsilon to prevent division by zero
        return ear

    def detect_blink(self, frame):
        # Update FPS
        self.resource_monitor.update_fps()
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)
        
        # Initialize frame data for JSON output
        frame_data = {
            "timestamp": time.time(),
            "total_blinks": self.total_blinks,
            "fps": self.resource_monitor.fps,
            "eye_state": self.previous_state,
            "ear_value": None,
            "face_detected": False
        }
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            frame_data["face_detected"] = True
            
            # Get eye coordinates
            left_eye_points = np.array([[landmarks[i].x * frame.shape[1], 
                                       landmarks[i].y * frame.shape[0]] 
                                      for i in self.LEFT_EYE])
            right_eye_points = np.array([[landmarks[i].x * frame.shape[1], 
                                        landmarks[i].y * frame.shape[0]] 
                                       for i in self.RIGHT_EYE])
            
            # Calculate EAR for both eyes
            left_ear = self.calculate_ear(left_eye_points)
            right_ear = self.calculate_ear(right_eye_points)
            
            # Average EAR
            avg_ear = (left_ear + right_ear) / 2.0
            frame_data["ear_value"] = avg_ear
            
            # Determine current state
            current_state = 'closed' if avg_ear < self.EAR_THRESHOLD else 'open'
            frame_data["eye_state"] = current_state
            
            # Blink detection logic with temporal consistency
            if current_state == 'closed' and self.previous_state == 'open':
                self.state_counter = 1
            elif current_state == 'closed' and self.previous_state == 'closed':
                self.state_counter += 1
            elif current_state == 'open' and self.previous_state == 'closed':
                if self.min_frames_closed <= self.state_counter <= self.max_frames_closed:
                    self.total_blinks += 1
                    frame_data["total_blinks"] = self.total_blinks
                self.state_counter = 0
            
            self.previous_state = current_state
            
            # Draw landmarks and eye regions
            for eye in [self.LEFT_EYE, self.RIGHT_EYE]:
                for i in eye:
                    x = int(landmarks[i].x * frame.shape[1])
                    y = int(landmarks[i].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
            
            # Draw status information
            cv2.putText(frame, f"Blinks: {self.total_blinks}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"State: {current_state}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {self.resource_monitor.fps:.1f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Output frame data as JSON
        json_output = json.dumps(frame_data, indent=2)
        print(json_output)
        
        return frame, self.total_blinks

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Eye Blink Tracker")
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
            }
            QLabel {
                color: #ffffff;
                font-size: 14px;
            }
        """)

        # Initialize the UI
        self.init_ui()
        
        # Initialize blink detector
        self.detector = BlinkDetector()
        
        # Initialize camera
        self.init_camera()
        
        # Start the video timer if camera is initialized
        if hasattr(self, 'cap') and self.cap is not None and self.cap.isOpened():
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)  # 30ms = ~33 fps

    def init_camera(self):
        # Try default camera
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            sys.exit(1)

    def init_ui(self):
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create video display label
        self.video_label = QLabel()
        self.video_label.setFixedSize(640, 480)
        layout.addWidget(self.video_label)
        
        # Create blink count label
        self.blink_count_label = QLabel("Total Blinks: 0")
        self.blink_count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.blink_count_label)

    def update_frame(self):
        if not hasattr(self, 'cap') or self.cap is None or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if ret:
            try:
                # Process frame for blink detection
                frame, total_blinks = self.detector.detect_blink(frame)
                
                # Update label with total blinks
                self.blink_count_label.setText(f"Total Blinks: {total_blinks}")
                
                # Convert frame to QImage for display
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                    self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
            except Exception as e:
                print(f"Error processing frame: {str(e)}")

    def closeEvent(self, event):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec()) 