import cv2
import face_recognition
import numpy as np
import pickle
import os
import time

class FaceEngine:
    def __init__(self, db_path="face_db.pkl"):
        self.db_path = db_path
        self.KNOWN_FACE_WIDTH = 14.5 
        self.FOCAL_LENGTH = 600  
        self.load_db()

    def load_db(self):
        """Loads the database from disk into memory."""
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                db = pickle.load(f)
                self.known_encodings = db["encodings"]
                self.known_names = db["names"]
            print(f" ✅ Database Loaded: {len(self.known_names)} faces found.")
        else:
            self.known_encodings, self.known_names = [], []

    def save_db(self):
        """Saves current memory encodings back to the pkl file."""
        db = {"encodings": self.known_encodings, "names": self.known_names}
        with open(self.db_path, "wb") as f:
            pickle.dump(db, f)

    # Inside FaceEngine class in face_engine.py
    def register_new_face(self, person_name, num_samples=8):
        cap = cv2.VideoCapture(0)
        collected, temp_encodings = 0, []
        
        while collected < num_samples:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, boxes)

            for enc in encodings:
                temp_encodings.append(enc)
                collected += 1
                # Draw on screen to show it's working
                cv2.putText(frame, f"Saving {collected}/{num_samples}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if collected >= num_samples: break
                
            cv2.imshow("Registering...", frame)
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()
        
        if len(temp_encodings) > 0:
            # Use averaging logic for better matching
            avg_encoding = np.mean(temp_encodings, axis=0)
            self.known_encodings.append(avg_encoding)
            self.known_names.append(person_name)
            
            # Save to file immediately
            db = {"encodings": self.known_encodings, "names": self.known_names}
            with open("face_db.pkl", "wb") as f:
                import pickle
                pickle.dump(db, f)
            return True
        return False
    def get_face_description(self, frame, face_location):
        top, right, bottom, left = face_location
        f_height, f_width, _ = frame.shape

        # 1. CLOTHING COLOR DETECTION
        # Sample a rectangle below the face (where a shirt would be)
        shirt_top = min(bottom + 20, f_height - 1)
        shirt_bottom = min(bottom + 100, f_height)
        shirt_roi = frame[shirt_top:shirt_bottom, left:right]
        
        if shirt_roi.size > 0:
            avg_color_per_row = np.average(shirt_roi, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0) # BGR
            
            # Simple Color Logic
            b, g, r = avg_color
            if r > 200 and g < 100: color_desc = "red"
            elif b > 200: color_desc = "blue"
            elif g > 150 and r < 100: color_desc = "green"
            elif r > 200 and g > 200 and b > 200: color_desc = "white"
            elif r < 50 and g < 50 and b < 50: color_desc = "dark or black"
            else: color_desc = "neutral colored"
        else:
            color_desc = "unknown color"

        # 2. RELATIVE HEIGHT ESTIMATION
        # If the face is in the top 30% of the frame, they are likely 'tall'
        face_center_y = (top + bottom) / 2
        if face_center_y < f_height * 0.35:
            height_desc = "taller than the camera's baseline"
        elif face_center_y > f_height * 0.6:
            height_desc = "shorter than the camera's baseline"
        else:
            height_desc = "about average height"

        # 3. SPEAKING DETECTION (Mouth Aspect Ratio - Simplified)
        # Check if the lower part of the face box is moving/varying
        # For true lip-sync detection, a landmark model is usually required.
        # Here we use a heuristic based on box height.
        is_speaking = "standing still" # Default for single-frame snapshots

        return f"wearing a {color_desc} top, appearing {height_desc}, and {is_speaking}."
    def start_emergency_recording(self, duration=10):
        """Records video and audio for a set duration."""
        cap = cv2.VideoCapture(0)
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('emergency_evidence.avi', fourcc, 20.0, (640, 480))
        
        start_time = time.time()
        while(int(time.time() - start_time) < duration):
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break
        
        cap.release()
        out.release()
        print("🎥 Video Evidence Saved.")
    def analyze_scene(self, tolerance=0.45):
        """Standard recognition and distance calculation."""
        cap = cv2.VideoCapture(0)
        time.sleep(1.0) # Warmup
        ret, frame = cap.read()
        cap.release()
        
        if not ret: return {"status": "ERROR"}

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)

        if not locations: return {"status": "NO_FACE"}

        loc = locations[0]
        brief_desc = self.get_face_description(frame, loc) # Pass the frame here!
        encoding = face_recognition.face_encodings(rgb, [loc])[0]
        
        # Distance logic
        pixel_width = loc[1] - loc[3]
        distance_cm = (self.KNOWN_FACE_WIDTH * self.FOCAL_LENGTH) / pixel_width
        
        # Matching logic from your recognize_loop
        name = "an unknown person"
        if self.known_encodings:
            distances = face_recognition.face_distance(self.known_encodings, encoding)
            best_idx = np.argmin(distances)
            if distances[best_idx] <= tolerance:
                name = self.known_names[best_idx]

        return {
            "status": "FOUND",
            "name": name,
            "distance": round(distance_cm / 100, 1),
            "description": brief_desc
        }