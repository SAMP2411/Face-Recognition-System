"""
Display utilities for face recognition visualization
"""
import cv2

def draw_face_info(frame, face, person_id, similarity, status="recognized", motion_info="", validation_info=""):
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    
    # Color coding
    colors = {
        "recognized": (0, 255, 0),
        "unknown": (0, 0, 255),
        "unknown_moving": (0, 165, 255),
        "unknown_stable": (255, 255, 0),
        "invalid": (128, 128, 128)
    }
    color = colors.get(status, (255, 255, 0))
    
    # Labels
    if status == "recognized":
        label = f"ID: {person_id}"
        confidence_text = f"Conf: {similarity:.3f}"
    elif status == "invalid":
        label = "INVALID FACE"
        confidence_text = validation_info
    else:
        label = "UNKNOWN"
        confidence_text = {"unknown": "New Person", "unknown_moving": "Wait for stability", 
                          "unknown_stable": "Ready for registration"}.get(status, "")
    
    # Draw
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    info_lines = [label, confidence_text, motion_info, validation_info]
    info_lines = [line for line in info_lines if line]
    
    label_bg_height = 25 * len(info_lines) + 10
    cv2.rectangle(frame, (x1, y1-label_bg_height), (x2, y1), color, -1)
    
    for i, line in enumerate(info_lines):
        line_y = y1 - label_bg_height + 20 + i*20
        font_size = 0.6 if i == 0 else 0.4
        cv2.putText(frame, line, (x1+5, line_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 2)
    
    # Confidence bar
    if status == "recognized" and similarity > 0:
        bar_width = int((similarity if similarity <= 1 else 1) * (x2-x1))
        cv2.rectangle(frame, (x1, y2+5), (x1 + bar_width, y2+15), color, -1)
        cv2.rectangle(frame, (x1, y2+5), (x2, y2+15), (128, 128, 128), 1)
    
    return frame

class TempFace:
    """Temporary face object for drawing"""
    def __init__(self, bbox):
        self.bbox = bbox