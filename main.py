"""
Main application - Face Recognition with Camera Display
Simplified version for local face recognition and registration
"""
import cv2
import time
from face_analyzer import FaceAnalyzer
from face_validator import FaceValidator, find_closest_valid_face
from unknown_tracker import UnknownFaceTracker
from data_manager import load_known_faces, log_event, generate_random_id
from faiss_manager import build_faiss_index, find_matching_person_fast
from capture_manager import capture_embeddings_for_person
from display_utils import draw_face_info, TempFace
from config import *

# ===== SIMPLIFIED FUNCTIONS =====

# GUI compatibility helpers: some OpenCV builds (headless) don't implement imshow/win
GUI_AVAILABLE = True

def try_show(window_name, frame):
    global GUI_AVAILABLE
    if not GUI_AVAILABLE:
        return
    try:
        cv2.imshow(window_name, frame)
    except Exception as e:
        print(f"⚠️  OpenCV GUI not available: {e}")
        GUI_AVAILABLE = False

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



def notify_person_detected(person_id, confidence, is_unknown, message, distance):
    """Log person detection (no API calls)"""
    if is_unknown:
        print(f"📢 [DETECTION] Unknown person detected - {message}")
    else:
        print(f"📢 [DETECTION] {person_id} - Confidence: {confidence:.4f}")
    return True

def notify_person_left_area():
    """Log person left area (no API calls)"""
    print(f"📢 [AREA] Person left detection area")
    return True

def notify_registration_start():
    """Log registration start (no API calls)"""
    print(f"📢 [REGISTRATION] Registration started")
    return True

def notify_registration_angle_instruction(angle_name, instruction, angle_number, total_angles):
    """Log registration angle instruction (no API calls)"""
    print(f"📢 [REGISTRATION] Angle {angle_number}/{total_angles}: {instruction}")
    return True

def notify_registration_complete(person_id, success=True, message=None):
    """Log registration complete (no API calls)"""
    print(f"📢 [REGISTRATION] Complete - {message or f'Registered {person_id}'}")
    return True

def notify_registration_failed(message="Registration failed", reason="Unknown"):
    """Log registration failed (no API calls)"""
    print(f"📢 [REGISTRATION] Failed - {message} ({reason})")
    return True

def check_user_registration_decision():
    """Check if user wants to register (simplified - auto-register unknown faces)"""
    return True, True

def test_frontend_connection():
    """Test connection (no API calls)"""
    print("✅ [LOCAL] Face Recognition System Ready")
    return True


# Enhanced capture using the capture_manager module


def main():
    """Main function with camera display - face recognition"""
    print("=" * 80)
    print("🏢 FACE RECOGNITION SYSTEM WITH CAMERA DISPLAY")
    print("🎯 Motion Detection + Distance Validation + Live Registration")
    print("🛑 Press 'q' or Ctrl+C to stop")
    print("=" * 80)
    
    # Test connection
    test_frontend_connection()
    print()
    
    # Initialize components
    face_analyzer = FaceAnalyzer()
    known_faces = load_known_faces()
    
    if known_faces:
        build_faiss_index(known_faces)
        print(f"📊 Loaded {len(known_faces)} known faces")
    else:
        print("📊 No known faces found - starting fresh")
    
    # Initialize camera (try multiple backends and indices)
    cap = None
    backends = [0, cv2.CAP_MSMF, cv2.CAP_DSHOW, -1]  # 0=auto, MSMF, DSHOW, auto again
    backend_names = ['auto', 'MSMF', 'DSHOW', 'auto']
    
    for backend, backend_name in zip(backends, backend_names):
        for camera_idx in [0, 1, -1]:
            try:
                if backend == 0 or backend == -1:
                    cap = cv2.VideoCapture(camera_idx)
                else:
                    cap = cv2.VideoCapture(camera_idx, backend)
                
                if cap.isOpened():
                    print(f"✅ Camera opened: backend={backend_name}, index={camera_idx}")
                    break
                else:
                    cap.release()
            except Exception as e:
                try:
                    cap.release()
                except Exception:
                    pass
        
        if cap and cap.isOpened():
            break
    
    if cap is None or not cap.isOpened():
        print("❌ Error: Camera not accessible with any backend or index.")
        return

    # Release and wait briefly before configuring to avoid kernel resource conflicts
    time.sleep(0.5)

    # Recognition persistence buffers
    recognition_buffer = None
    recognition_count = 0
    recognition_similarity_acc = 0.0
        
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    
    unknown_tracker = UnknownFaceTracker()
    
    # Statistics
    stats = {"recognitions": 0, "registrations": 0, "duplicates": 0, "motion_prevented": 0, "invalid_faces": 0}
    last_stats_print = time.time()
     
    last_detected_id = None
    last_detection_time = 0
    frame_count = 0
    person_in_area = False
    
    # Anti-flicker variables
    last_face_info = None
    no_face_count = 0
    
    # Camera error recovery
    consecutive_read_failures = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                consecutive_read_failures += 1
                # If camera is consistently failing, attempt to recover
                if consecutive_read_failures > 30:
                    print(f"⚠️  Too many read failures ({consecutive_read_failures}), attempting camera recovery...")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    time.sleep(1.0)
                    # Try to reopen with the same backend
                    cap = cv2.VideoCapture(0)
                    if cap.isOpened():
                        print("✅ Camera recovered")
                        consecutive_read_failures = 0
                    else:
                        print("❌ Camera recovery failed, continuing...")
                        consecutive_read_failures = 0
                continue
            
            # Success, reset failure counter
            consecutive_read_failures = 0

            frame_count += 1
            current_time = time.time()
            # Brighten and slightly boost contrast for smoother detection
            frame = cv2.convertScaleAbs(frame, alpha=1.08, beta=18)
            faces = face_analyzer.get_faces(frame)
            
            # Print stats periodically
            if current_time - last_stats_print >= 30:
                print(f"📊 [STATS] Known: {len(known_faces)} | Recognitions: {stats['recognitions']} | "
                      f"Registrations: {stats['registrations']} | Motion Prevented: {stats['motion_prevented']}")
                last_stats_print = current_time
            
            # Draw stats on camera feed
            info_lines = [
                f"Face Recognition + Camera Display",
                f"Known People: {len(known_faces)} | Recognitions: {stats['recognitions']}",
                f"Registrations: {stats['registrations']} | Duplicates Prevented: {stats['duplicates']}",
                f"Motion Prevented: {stats['motion_prevented']} | Out of Range: {stats['invalid_faces']}"
            ]
            
            for i, line in enumerate(info_lines):
                font_size = 0.5 if i == 0 else 0.4
                cv2.putText(frame, line, (10, 25 + i*18), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), 1)
            
            # Process faces
            if faces:
                face = find_closest_valid_face(faces, frame.shape)
                
                if face is None:
                    # All faces out of range
                    stats["invalid_faces"] += 1 if frame_count % 30 == 0 else 0
                    no_face_count += 1
                    
                    if no_face_count > FACE_PERSISTENCE:
                        if person_in_area:
                            notify_person_left_area()
                            person_in_area = False
                        unknown_tracker.reset()
                        last_face_info = None
                    
                    cv2.putText(frame, "Move closer to camera (within 50cm)", 
                               (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                else:
                    # Valid face within range
                    no_face_count = 0
                    person_in_area = True
                    is_valid, validation_info, distance = FaceValidator.validate_face(face, frame.shape)
                    
                    # Process every 3rd frame for performance
                    if frame_count % 3 == 0:
                        person_id, similarity = find_matching_person_fast(face.embedding, SIMILARITY_THRESHOLD)

                        if person_id:
                            # Buffer recognition to avoid single-frame false positives
                            if recognition_buffer == person_id:
                                recognition_count += 1
                                recognition_similarity_acc += similarity
                            else:
                                recognition_buffer = person_id
                                recognition_count = 1
                                recognition_similarity_acc = similarity

                            # Confirm after 2 consistent detections
                            if recognition_count >= 2:
                                avg_similarity = recognition_similarity_acc / recognition_count
                                # Known person recognized
                                last_face_info = {
                                    "type": "recognized",
                                    "person_id": person_id,
                                    "similarity": avg_similarity,
                                    "validation_info": validation_info,
                                    "bbox": face.bbox
                                }
                                unknown_tracker.reset()

                                # Throttle notifications (2 seconds minimum between same person)
                                if person_id != last_detected_id or (current_time - last_detection_time) > 2:
                                    print(f"👋 [RECOGNIZED] Welcome back, {person_id}! (Similarity: {avg_similarity:.4f}, Distance: {distance:.0f}cm)")

                                    # Notify
                                    notify_person_detected(
                                        person_id=person_id,
                                        confidence=avg_similarity,
                                        is_unknown=False,
                                        message=f"Welcome back, {person_id}!",
                                        distance=distance
                                    )

                                    stats["recognitions"] += 1
                                    last_detected_id = person_id
                                    last_detection_time = current_time
                                    log_event("recognition", person_id, avg_similarity)
                                # reset buffer after confirmation to avoid repeated triggers
                                recognition_buffer = None
                                recognition_count = 0
                                recognition_similarity_acc = 0.0
                        else:
                            # Unknown person
                            should_register, status_message, _ = unknown_tracker.track_unknown_face(
                                face.embedding, frame, face.bbox.astype(int), current_time)
                            
                            # Visual status
                            if "MOVING" in status_message:
                                visual_status = "unknown_moving"
                                if frame_count % 90 == 0:
                                    print("🔄 Unknown person moving - waiting for stability...")
                                    stats["motion_prevented"] += 1
                            elif "stable" in status_message.lower() and should_register:
                                visual_status = "unknown_stable"
                            else:
                                visual_status = "unknown"
                            
                            motion_info = status_message.split('|')[1].strip() if '|' in status_message else ""
                            
                            last_face_info = {
                                "type": visual_status,
                                "person_id": None,
                                "similarity": 0,
                                "validation_info": validation_info,
                                "motion_info": motion_info,
                                "bbox": face.bbox,
                                "status_message": status_message
                            }
                            
                            if should_register:
                                print("🆕 [STABLE] Unknown person detected - starting auto-registration...")
                                
                                # Notify
                                notify_person_detected(
                                    person_id=None,
                                    confidence=0,
                                    is_unknown=True,
                                    message="Unknown person detected. Registering...",
                                    distance=distance
                                )
                                
                                # Generate unique person ID
                                person_id = generate_random_id()
                                while person_id in known_faces:
                                    person_id = generate_random_id()
                                
                                # Show registration starting message on camera
                                cv2.putText(frame, "STARTING REGISTRATION - FOLLOW INSTRUCTIONS", 
                                           (10, frame.shape[0]-80), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                                try_show("Face Recognition System", frame)
                                try_waitkey(2000)
                                
                                # Perform registration using capture_embeddings_for_person
                                print(f"🎯 Starting registration for {person_id}")
                                notify_registration_start()
                                
                                # Reload known faces before registration
                                known_faces = load_known_faces()
                                registration_successful = capture_embeddings_for_person(person_id, known_faces)
                                
                                if registration_successful:
                                    # Rebuild index with new face
                                    known_faces = load_known_faces()
                                    build_faiss_index(known_faces)
                                    print(f"✅ [SUCCESS] Person {person_id} registered successfully!")
                                    notify_registration_complete(person_id, True, f"Registered {person_id}")
                                    
                                    stats["registrations"] += 1
                                    log_event("registration", person_id)
                                    last_detected_id = person_id
                                    last_detection_time = time.time()
                                else:
                                    print(f"❌ [FAILED] Registration failed - duplicate or cancelled")
                                    notify_registration_failed("Registration failed", "duplicate_or_cancelled")
                                    stats["duplicates"] += 1
                                
                                unknown_tracker.reset()
                                last_face_info = None
                                continue
            else:
                # No faces detected
                no_face_count += 1
                
                if no_face_count > FACE_PERSISTENCE * 2:
                    if person_in_area:
                        notify_person_left_area()
                        person_in_area = False
                    unknown_tracker.reset()
                    last_face_info = None
                    cv2.putText(frame, "No faces detected - position yourself in front of camera", 
                               (10, frame.shape[0]-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw persistent face info (anti-flicker) on camera feed
            if last_face_info:
                temp_face = TempFace(last_face_info["bbox"])
                
                if last_face_info["type"] == "recognized":
                    frame = draw_face_info(frame, temp_face, last_face_info["person_id"], 
                                         last_face_info["similarity"], "recognized", "", 
                                         last_face_info["validation_info"])
                else:
                    frame = draw_face_info(frame, temp_face, None, 0, last_face_info["type"], 
                                         last_face_info.get("motion_info", ""), 
                                         last_face_info["validation_info"])
                    
                    # Show status messages on camera
                    if "status_message" in last_face_info:
                        status_lines = last_face_info["status_message"].split('|')
                        for i, line in enumerate(status_lines):
                            cv2.putText(frame, line.strip(), (10, frame.shape[0] - 100 + i*20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Show camera feed
            try_show("Face Recognition System", frame)
            
            # Exit condition
            if try_waitkey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    finally:
        # Cleanup
        if person_in_area:
            notify_person_left_area()
        
        cap.release()
        try_destroy_all()
        
        print("\n✅ Session Summary:")
        for key, value in stats.items():
            print(f"   - {key.replace('_', ' ').title()}: {value}")
        print(f"   - Known People: {len(load_known_faces())}")


if __name__ == '__main__':
    print("🚀 Starting Face Recognition System with Camera Display")
    print()
    
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
