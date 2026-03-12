import cv2
import face_recognition
import numpy as np
import pickle
import os


class FaceRecognitionModule:

    def __init__(self, db_path=None):

        # Ensure database is stored in project models folder
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

        if db_path is None:
            db_path = os.path.join(PROJECT_ROOT, "models", "face_db.pkl")

        self.db_path = db_path

        self.KNOWN_FACE_WIDTH = 14.5
        self.FOCAL_LENGTH = 600

        self.load_db()

    # ---------------- LOAD DATABASE ----------------

    def load_db(self):

        if os.path.exists(self.db_path):

            with open(self.db_path, "rb") as f:
                db = pickle.load(f)

                self.known_encodings = db["encodings"]
                self.known_names = db["names"]

            print(f"✅ Face database loaded ({len(self.known_names)} faces)")

        else:
            self.known_encodings = []
            self.known_names = []

    # ---------------- SAVE DATABASE ----------------

    def save_db(self):

        with open(self.db_path, "wb") as f:

            pickle.dump(
                {
                    "encodings": self.known_encodings,
                    "names": self.known_names
                },
                f
            )

    # ---------------- REGISTER NEW FACE ----------------

    def register_new_face(self, person_name, num_samples=8):

        cap = cv2.VideoCapture(0)

        collected = 0
        temp_encodings = []

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

                cv2.putText(
                    frame,
                    f"Saving {collected}/{num_samples}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Register Face", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        if temp_encodings:

            avg_encoding = np.mean(temp_encodings, axis=0)

            self.known_encodings.append(avg_encoding)
            self.known_names.append(person_name)

            self.save_db()

            return True

        return False

    # ---------------- IDENTIFY FACE ----------------

    def identify_face(self, encoding):

        if not self.known_encodings:
            return "Unknown"

        distances = face_recognition.face_distance(self.known_encodings, encoding)

        idx = np.argmin(distances)

        if distances[idx] <= 0.45:
            return self.known_names[idx]

        return "Unknown"


# =========================================================
# PUBLIC FUNCTIONS FOR MODULE4
# =========================================================

def register_face(person_name):

    recognizer = FaceRecognitionModule()

    success = recognizer.register_new_face(person_name)

    if success:
        return f"{person_name} has been registered successfully."

    return "Face registration failed."


def recognize_face():

    recognizer = FaceRecognitionModule()

    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    cap.release()

    if not ret:
        return "Camera capture failed."

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb)

    encodings = face_recognition.face_encodings(rgb, boxes)

    if not encodings:
        return "I do not see any face."

    name = recognizer.identify_face(encodings[0])

    if name == "Unknown":
        return "I do not recognize this person."

    return f"{name} is in front of you."