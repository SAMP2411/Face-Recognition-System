"""
Motion detection for face stability analysis
"""
import cv2
import numpy as np
from config import MOTION_THRESHOLD, MOTION_HISTORY_SIZE

class MotionDetector:
    def __init__(self):
        self.motion_threshold = MOTION_THRESHOLD
        self.history_size = MOTION_HISTORY_SIZE
        self.previous_face_region = None
        self.motion_history = []
        self.last_bbox = None
        
    def detect_face_motion(self, frame, face_bbox):
        try:
            x1, y1, x2, y2 = [max(0, min(frame.shape[1] if i%2==0 else frame.shape[0], coord)) 
                             for i, coord in enumerate(face_bbox)]
            
            if x2 <= x1 or y2 <= y1:
                return True, "Invalid face region", 0
            
            current_gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
            
            # Calculate bbox motion - reduced sensitivity
            bbox_motion = 0
            if self.last_bbox is not None:
                prev_center = ((self.last_bbox[0] + self.last_bbox[2]) / 2, 
                              (self.last_bbox[1] + self.last_bbox[3]) / 2)
                curr_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                bbox_motion = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                                    (curr_center[1] - prev_center[1])**2)
            
            self.last_bbox = (x1, y1, x2, y2)
            
            # Calculate pixel motion - reduced sensitivity
            pixel_motion = 0
            if self.previous_face_region is not None:
                if current_gray.shape != self.previous_face_region.shape:
                    current_gray = cv2.resize(current_gray, self.previous_face_region.shape[::-1])
                
                diff = cv2.absdiff(current_gray, self.previous_face_region)
                _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                pixel_motion = np.sum(thresh > 0)
            
            self.previous_face_region = current_gray.copy()
            
            # Reduced weight for bbox motion
            total_motion = pixel_motion + (bbox_motion * 50)
            self.motion_history.append(total_motion)
            if len(self.motion_history) > self.history_size:
                self.motion_history.pop(0)
            
            # Use median for more stable motion detection
            avg_motion = np.median(self.motion_history) if len(self.motion_history) > 3 else total_motion
            is_moving = avg_motion > self.motion_threshold
            
            motion_info = f"Motion: {avg_motion:.0f} ({'MOVING' if is_moving else 'STABLE'})"
            if bbox_motion > 0:
                motion_info += f" | Head: {bbox_motion:.1f}px"
            
            return is_moving, motion_info, avg_motion
            
        except Exception as e:
            return True, f"Motion error: {e}", 0
    
    def reset(self):
        self.previous_face_region = None
        self.motion_history = []
        self.last_bbox = None