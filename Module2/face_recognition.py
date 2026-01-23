import cv2
import face_recognition
import pickle
import os
import numpy as np
import time

DB_PATH = "face_db.pkl"

def load_database():
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {"encodings": [], "names": []}
    return {"encodings": [], "names": []}

def save_database(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)

def clean_image_for_dlib(frame):
    """
    NUCLEAR OPTION: Creates a fresh memory block to satisfy Dlib.
    """
    try:
        # 1. Handle Empty
        if frame is None: return None
        
        # 2. Force standard NumPy array
        frame = np.array(frame)
        
        # 3. Get Dimensions
        h, w = frame.shape[:2]
        
        # 4. Create a BRAND NEW blank image (Guaranteed Clean Memory)
        clean_frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # 5. Copy data manually (This strips out Alpha channels/Bad Strides)
        # We only take the first 3 channels (BGR)
        if len(frame.shape) == 3:
            clean_frame[:, :, :] = frame[:, :, :3]
        else:
            # If grayscale, replicate channels
            clean_frame[:, :, 0] = frame
            clean_frame[:, :, 1] = frame
            clean_frame[:, :, 2] = frame
            
        # 6. Convert BGR to RGB
        rgb = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
        
        return rgb
    except Exception as e:
        print(f"[IMG CLEAN ERROR] {e}")
        return None

def register_person(db, person_name, num_samples=8):
    print(f"[INFO] Opening Camera to register {person_name}...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("[ERROR] Could not open any camera.")
            return False

    time.sleep(1.0) # Warmup
    collected = 0
    temp_encodings = []
    
    print(f"[INFO] Starting capture. Look at the camera.")

    while collected < num_samples:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.1)
            continue

        # --- USE THE NUCLEAR CLEANER ---
        rgb = clean_image_for_dlib(frame)
        
        if rgb is None:
            continue

        try:
            # Now Dlib should be happy
            boxes = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, boxes)

            # Draw on original frame
            for (top, right, bottom, left), enc in zip(boxes, encodings):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                if collected < num_samples:
                    temp_encodings.append(enc)
                    collected += 1
                    print(f"   -> Captured Sample {collected}/{num_samples}")

            cv2.putText(frame, f"Collected: {collected}/{num_samples}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception as e:
            print(f"[CRASH] Dlib rejected the image: {e}")
            continue

        cv2.imshow("Registering Face", frame)
        if boxes: time.sleep(0.5)
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    cap.release()
    cv2.destroyAllWindows()
    
    if len(temp_encodings) > 0:
        avg_encoding = np.mean(temp_encodings, axis=0)
        db["encodings"].append(avg_encoding)
        db["names"].append(person_name)
        save_database(db)
        print(f"[SUCCESS] Registered {person_name}.")
        return True
    else:
        print("[FAIL] No face data collected.")
        return False

def recognize_single_frame(db):
    """Captures ONE frame and returns the name."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): return "Camera Error"

    # Warmup
    for _ in range(5): cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None: return "Camera Error"

    # Clean Image
    rgb = clean_image_for_dlib(frame)
    if rgb is None: return "Camera Error"

    known_encodings = db["encodings"]
    known_names = db["names"]

    if not known_encodings: return "NO_DB"

    try:
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)
        found_names = []
        
        for enc in encodings:
            distances = face_recognition.face_distance(known_encodings, enc)
            if len(distances) > 0:
                best_idx = np.argmin(distances)
                if distances[best_idx] <= 0.45:
                    found_names.append(known_names[best_idx])
        
        if not found_names: return "UNKNOWN"
        return ", ".join(list(set(found_names)))
        
    except Exception:
        return "Camera Error"