"""
Unknown face tracking module
"""
import time
import numpy as np
from scipy.spatial.distance import cosine
from motion_detector import MotionDetector
from config import STABILITY_TIME, REGISTRATION_DELAY

class UnknownFaceTracker:
    def __init__(self):
        self.motion_detector = MotionDetector()
        self.reset()
        
    def reset(self):
        self.first_detection_time = None
        self.last_embedding = None
        self.consistent_detections = 0
        self.stability_start_time = None
        self.is_ready_for_registration = False
        self.motion_detector.reset()
    
    def is_same_face(self, embedding1, embedding2, threshold=0.8):
        try:
            return (1 - cosine(embedding1, embedding2)) > threshold
        except:
            return False
    
    def track_unknown_face(self, face_embedding, frame, face_bbox, current_time):
        is_moving, motion_info, _ = self.motion_detector.detect_face_motion(frame, face_bbox)
        
        if self.first_detection_time is None:
            self.first_detection_time = current_time
            self.last_embedding = face_embedding.copy()
            self.consistent_detections = 1
            self.stability_start_time = None
            return False, f"🔍 New unknown face detected | {motion_info}", 0
        
        if not self.is_same_face(face_embedding, self.last_embedding):
            self.reset()
            self.first_detection_time = current_time
            self.last_embedding = face_embedding.copy()
            self.consistent_detections = 1
            return False, f"🔍 New unknown face detected | {motion_info}", 0
        
        self.consistent_detections += 1
        self.last_embedding = face_embedding.copy()
        
        if is_moving:
            self.stability_start_time = None
            time_since_first = current_time - self.first_detection_time
            return False, f"🚫 Face moving - waiting for stability | {motion_info} | Detected: {time_since_first:.1f}s", 0
        else:
            if self.stability_start_time is None:
                self.stability_start_time = current_time
            
            stability_duration = current_time - self.stability_start_time
            
            if stability_duration >= STABILITY_TIME:
                total_time = current_time - self.first_detection_time
                
                if total_time >= REGISTRATION_DELAY and self.consistent_detections >= 10:
                    return True, f"✅ Face stable - starting registration | {motion_info}", 0
                else:
                    time_remaining = max(0, REGISTRATION_DELAY - total_time)
                    return False, f"⏱️ Face stable - waiting {time_remaining:.1f}s | {motion_info}", time_remaining
            else:
                stability_remaining = STABILITY_TIME - stability_duration
                return False, f"⏸️ Stabilizing... {stability_remaining:.1f}s | {motion_info}", stability_remaining