"""
Safe display compatibility helpers for OpenCV GUI-less builds
"""
import cv2

GUI_AVAILABLE = True


def try_show(window_name, frame):
    global GUI_AVAILABLE
    if not GUI_AVAILABLE:
        return
    try:
        cv2.imshow(window_name, frame)
    except Exception as e:
        # mark as unavailable to avoid repeated exceptions
        GUI_AVAILABLE = False
        print(f"⚠️  OpenCV GUI not available: {e}")


def try_waitkey(delay=1):
    if not GUI_AVAILABLE:
        return -1
    try:
        return cv2.waitKey(delay)
    except Exception:
        return -1


def try_destroy_all():
    if not GUI_AVAILABLE:
        return
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
