import cv2
import face_recognition
import numpy as np
import pickle
import os

class FaceRecognitionModule:
    def __init__(self, db_path="face_db.pkl"):
        self.db_path = db_path
        self.KNOWN_FACE_WIDTH = 14.5
        self.FOCAL_LENGTH = 600
        self.load_db()

    def load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, "rb") as f:
                db = pickle.load(f)
                self.known_encodings = db["encodings"]
                self.known_names = db["names"]
            print(f"✅ Database Loaded: {len(self.known_names)} faces found.")
        else:
            self.known_encodings, self.known_names = [], []

    def save_db(self):
        with open(self.db_path, "wb") as f:
            pickle.dump({"encodings": self.known_encodings, "names": self.known_names}, f)

    def register_new_face(self, person_name, num_samples=8):
        cap = cv2.VideoCapture(0)
        collected, temp_encodings = 0, []
        print(f"📸 Registering {person_name}...")
        while collected < num_samples:
            ret, frame = cap.read()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, boxes)
            for enc in encodings:
                temp_encodings.append(enc)
                collected += 1
                cv2.putText(frame, f"Saving {collected}/{num_samples}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Registering Face", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()
        if temp_encodings:
            self.known_encodings.append(np.mean(temp_encodings, axis=0))
            self.known_names.append(person_name)
            self.save_db()
            return True
        return False

    def identify_face(self, encoding):
        if not self.known_encodings: return "Unknown"
        distances = face_recognition.face_distance(self.known_encodings, encoding)
        idx = np.argmin(distances)
        return self.known_names[idx] if distances[idx] <= 0.45 else "Unknown"

# --- INDEPENDENT EXECUTION ---
if __name__ == "__main__":
    recognizer = FaceRecognitionModule()
    print("Press 'r' to register a new face, 'q' to quit.")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret: break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locs = face_recognition.face_locations(rgb)
        encs = face_recognition.face_encodings(rgb, locs)
        for loc, enc in zip(locs, encs):
            name = recognizer.identify_face(enc)
            cv2.rectangle(frame, (loc[3], loc[0]), (loc[1], loc[2]), (0, 255, 0), 2)
            cv2.putText(frame, name, (loc[3], loc[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Face Recognition Mode", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            cap.release()
            cv2.destroyAllWindows()
            name_input = input("Enter name: ")
            recognizer.register_new_face(name_input)
            cap = cv2.VideoCapture(0)
    cap.release()
    cv2.destroyAllWindows()