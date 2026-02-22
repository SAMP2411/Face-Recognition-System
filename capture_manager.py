"""
Face capture and registration management
"""
import cv2
import time
import numpy as np
from tqdm import tqdm
from face_analyzer import FaceAnalyzer
from face_validator import FaceValidator
from data_manager import save_known_faces
from faiss_manager import check_for_duplicate_during_registration
from config import ANGLE_INSTRUCTIONS, EMBEDDINGS_PER_ANGLE, THRESHOLD, DUPLICATE_THRESHOLD
from collections import deque
from display_compat import try_show, try_waitkey, try_destroy_all

def capture_embeddings_for_person(person_id, known_faces):
    # Open camera (try multiple backends and indices)
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
        return False

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    cap.set(cv2.CAP_PROP_FPS, 15)

    app = FaceAnalyzer().app  # Get the InsightFace app
    embeddings = []
    print(f"📸 Starting capture for Person ID: {person_id}")

    for instruction, expected_yaw in ANGLE_INSTRUCTIONS:
        print(f"[INFO] Please: {instruction}")
        time.sleep(2)
        captured = 0
        pbar = tqdm(total=EMBEDDINGS_PER_ANGLE, desc=f"Capturing: {instruction}")

        # Wait for pose (with smoothing, brightness and distance validation)
        turned = False
        hold_start = None
        read_failures = 0
        yaw_history = deque(maxlen=5)
        while not turned:
            ret, frame = cap.read()
            if not ret:
                read_failures += 1
                if read_failures > 10:
                    print("⚠️ Too many failed reads while waiting for pose; attempting to reopen camera")
                    try:
                        cap.release()
                    except Exception:
                        pass
                    time.sleep(0.5)
                    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                    read_failures = 0
                    if not cap.isOpened():
                        print("❌ Failed to reopen camera")
                        return False
                continue

            # Brighten frame slightly to improve detection in poor lighting
            frame = cv2.convertScaleAbs(frame, alpha=1.15, beta=25)

            faces = app.get(frame)
            if not faces:
                continue

            face = faces[0]
            yaw, _ = FaceValidator.calculate_pose_angles(face)
            if yaw is not None:
                yaw_history.append(yaw)

            # Use median yaw to smooth small jitters
            median_yaw = None
            if len(yaw_history) >= 3:
                sorted_y = sorted(yaw_history)
                median_y = sorted_y[len(sorted_y)//2]
                median_yaw = float(median_y)
            else:
                median_yaw = yaw if yaw is not None else None

            if median_yaw is not None and abs(median_yaw - expected_yaw) < THRESHOLD:
                if hold_start is None:
                    hold_start = time.time()
                elif time.time() - hold_start > 1.2:
                    turned = True
            else:
                hold_start = None

            # Show preview
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 2)
            cv2.putText(frame, f"Please: {instruction}", (bbox[0], bbox[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            try_show("Registration", frame)
            try_waitkey(1)

        # Capture embeddings
        while captured < EMBEDDINGS_PER_ANGLE:
            ret, frame = cap.read()
            if not ret:
                read_failures += 1
                if read_failures > 10:
                    print("⚠️ Too many failed reads during capture; aborting registration")
                    cap.release()
                    try_destroy_all()
                    return False
                continue
            read_failures = 0

            faces = app.get(frame)
            if not faces:
                continue

            face = faces[0]
            yaw, _ = FaceValidator.calculate_pose_angles(face)
            
            if yaw is not None and abs(yaw - expected_yaw) < THRESHOLD:
                # Brighten for stability
                frame = cv2.convertScaleAbs(frame, alpha=1.12, beta=20)

                # Check for duplicates with configured threshold
                is_duplicate, matched_id, similarity = check_for_duplicate_during_registration(
                    face.embedding, known_faces, DUPLICATE_THRESHOLD)
                
                if is_duplicate:
                    pbar.close()
                    cap.release()
                    try_destroy_all()
                    
                    # Show duplicate message
                    duplicate_frame = np.zeros((300, 800, 3), dtype=np.uint8)
                    cv2.putText(duplicate_frame, "DUPLICATE DETECTED!", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.putText(duplicate_frame, f"Existing ID: {matched_id}", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(duplicate_frame, "Press any key to continue...", (50, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 2)
                    try_show("Duplicate Detection", duplicate_frame)
                    try_waitkey(0)
                    try_destroy_all()
                    return False
                
                embeddings.append(face.embedding)
                captured += 1
                pbar.update(1)
                
                # Visual feedback
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 5)
                cv2.putText(frame, "✅ CAPTURED!", (bbox[0], bbox[1]-50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                try_show("Registration", frame)
                try_waitkey(200)

        pbar.close()

    cap.release()
    try_destroy_all()

    known_faces[person_id] = embeddings
    save_known_faces(known_faces)
    print(f"✅ SUCCESS: {len(embeddings)} embeddings saved for {person_id}")
    return True