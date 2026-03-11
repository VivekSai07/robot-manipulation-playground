import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import urllib.request

class MediaPipeTeleop:
    """
    Mark-10 Spatial Teleoperation Interface (Modern Tasks API).
    Uses a webcam and MediaPipe to track the human hand in 3D space.
    - Thumb + Index Pinch: Close Gripper
    - Index + Middle Pinch: Enable Movement (Clutch)
    """
    def __init__(self, speed_scale=75.0, smoothing=0.6):
        # Automatically download the required model file if missing
        self.model_path = 'hand_landmarker.task'
        if not os.path.exists(self.model_path):
            print("🌐 Downloading MediaPipe Hand Landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, self.model_path)
            print("✅ Download complete!")

        # Setup the Modern Tasks API
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("❌ Could not open webcam!")

        self.v_des = np.zeros(6)
        self.gripper_closed = False
        self.reset_requested = False
        
        self.speed_scale = speed_scale
        self.smoothing = smoothing
        
        self.reference_palm = None
        self.last_detection_time = time.time()

        self.connections = [
            (0,1), (1,2), (2,3), (3,4), (0,5), (5,6), (6,7), (7,8), 
            (5,9), (9,10), (10,11), (11,12), (9,13), (13,14), (14,15), 
            (15,16), (13,17), (17,18), (18,19), (19,20), (0,17)
        ]

    def get_command(self):
        success, img = self.cap.read()
        if not success:
            return self.v_des.copy(), self.gripper_closed

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = self.detector.detect(mp_image)
        
        target_v = np.zeros(6)
        hand_detected = False
        movement_enabled = False

        if results.hand_landmarks:
            hand_detected = True
            self.last_detection_time = time.time()
            hand_lms = results.hand_landmarks[0]
            
            # --- 1. THE CLUTCH (Index + Middle Pinch) ---
            index_tip = np.array([hand_lms[8].x, hand_lms[8].y, hand_lms[8].z])
            middle_tip = np.array([hand_lms[12].x, hand_lms[12].y, hand_lms[12].z])
            clutch_dist = np.linalg.norm(index_tip - middle_tip)
            
            # If fingers are close, movement is active!
            movement_enabled = clutch_dist < 0.05
            
            # --- 2. SPATIAL MOVEMENT (Palm Tracking) ---
            palm_x = hand_lms[9].x
            palm_y = hand_lms[9].y
            palm_z = hand_lms[9].z 
            current_palm = np.array([palm_x, palm_y, palm_z])

            if self.reference_palm is None or not movement_enabled:
                # If paused, update reference so it doesn't jump when unpaused!
                self.reference_palm = current_palm
            else:
                delta = current_palm - self.reference_palm
                self.reference_palm = current_palm 
                
                target_v[1] = -delta[0] * self.speed_scale
                target_v[2] = -delta[1] * self.speed_scale
                target_v[0] = delta[2] * self.speed_scale * 1.5 

            # --- 3. GRIPPER CONTROL (Thumb + Index Pinch) ---
            thumb_tip = np.array([hand_lms[4].x, hand_lms[4].y, hand_lms[4].z])
            pinch_dist = np.linalg.norm(thumb_tip - index_tip)
            
            if pinch_dist < 0.05:
                self.gripper_closed = True
            elif pinch_dist > 0.08:
                self.gripper_closed = False

            # --- 4. MANUAL SKELETON DRAWING ---
            h, w, c = img.shape
            for connection in self.connections:
                idx1, idx2 = connection
                pt1 = (int(hand_lms[idx1].x * w), int(hand_lms[idx1].y * h))
                pt2 = (int(hand_lms[idx2].x * w), int(hand_lms[idx2].y * h))
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)
                
            for lm in hand_lms:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 4, (255, 0, 0), cv2.FILLED)
            
            # Visual feedback for Gripper
            if self.gripper_closed:
                cx, cy = int(hand_lms[8].x * w), int(hand_lms[8].y * h)
                cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED)

        # UI Overlay for the Clutch State
        if movement_enabled:
            cv2.putText(img, "MOVING", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        else:
            cv2.putText(img, "PAUSED (Pinch Index+Middle to move)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Safety Reset
        if not hand_detected and (time.time() - self.last_detection_time > 1.0):
            self.reference_palm = None

        # Smoothing
        self.v_des += (target_v - self.v_des) * self.smoothing

        cv2.imshow("Mark-10 Spatial Teleop", img)
        cv2.waitKey(1)

        return self.v_des.copy(), self.gripper_closed

    def cleanup(self):
        self.cap.release()
        cv2.destroyAllWindows()