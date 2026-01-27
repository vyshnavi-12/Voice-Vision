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
            pickle.dump({
                "encodings": self.known_encodings,
                "names": self.known_names
            }, f)

    def register_new_face(self, person_name, num_samples=8):
        cap = cv2.VideoCapture(0)
        collected, temp_encodings = 0, []

        print(f"📸 Registering {person_name}...")

        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, boxes)

            for enc in encodings:
                temp_encodings.append(enc)
                collected += 1
                cv2.putText(frame, f"Saving {collected}/{num_samples}",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
                if collected >= num_samples:
                    break

            cv2.imshow("Registering Face", frame)
            cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

        if temp_encodings:
            avg_encoding = np.mean(temp_encodings, axis=0)
            self.known_encodings.append(avg_encoding)
            self.known_names.append(person_name)
            self.save_db()
            print("✅ Face registered successfully")
            return True

        return False

    def get_face_description(self, frame, loc):
        top, right, bottom, left = loc
        h, _, _ = frame.shape

        face_y = (top + bottom) / 2
        if face_y < h * 0.35:
            height = "tall"
        elif face_y > h * 0.6:
            height = "short"
        else:
            height = "average height"

        return height

    def run(self):
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("FaceEngine")  # forces window creation
        print("▶ FaceEngine running")
        print("Controls: q=quit | r=register unknown | n=ignore")

        awaiting_decision = False
        last_unknown_face = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb)
            encs = face_recognition.face_encodings(rgb, locs)

            for loc, enc in zip(locs, encs):
                top, right, bottom, left = loc
                name = "Unknown"

                if self.known_encodings:
                    distances = face_recognition.face_distance(self.known_encodings, enc)
                    idx = np.argmin(distances)
                    if distances[idx] <= 0.45:
                        name = self.known_names[idx]
                        awaiting_decision = False

                if name == "Unknown":
                    awaiting_decision = True
                    last_unknown_face = True

                    cv2.putText(frame,
                                "Unknown face detected | Press R to register, N to ignore",
                                (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2)

                pixel_width = right - left
                distance = (self.KNOWN_FACE_WIDTH * self.FOCAL_LENGTH) / pixel_width
                distance_m = round(distance / 100, 2)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame,
                            f"{name} | {distance_m}m",
                            (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

            cv2.imshow("FaceEngine", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('r') and awaiting_decision:
                cap.release()
                cv2.destroyAllWindows()

                person_name = input("Enter person name to register: ")
                self.register_new_face(person_name)

                cap = cv2.VideoCapture(0)
                cv2.namedWindow("FaceEngine")
                awaiting_decision = False

            elif key == ord('n'):
                awaiting_decision = False

        cap.release()
        cv2.destroyAllWindows()



# =====================
# ENTRY POINT
# =====================
if __name__ == "__main__":
    engine = FaceEngine()
    engine.run()
