import cv2
import face_recognition
import numpy as np
import pickle
import os


# This class stores and recognizes faces
class FaceRecognitionModule:
    # Set up the face recognition system and load saved faces from database
    def __init__(self, db_path=None):

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

        if db_path is None:
            db_path = os.path.join(PROJECT_ROOT, "models", "face_db.pkl")

        self.db_path = db_path

        self.KNOWN_FACE_WIDTH = 14.5
        self.FOCAL_LENGTH = 600

        self.load_db()

    # Load all saved face data from file
    def load_db(self):

        if os.path.exists(self.db_path):

            with open(self.db_path, "rb") as f:

                db = pickle.load(f)

                self.known_encodings = db["encodings"]
                self.known_names = db["names"]

            print(f"✅ Face database loaded ({len(self.known_names)} faces)")

        else:
            # Start with empty lists if no database exists yet
            self.known_encodings = []
            self.known_names = []

    # Save all face data to a file for later use
    def save_db(self):
        with open(self.db_path, "wb") as f:

            pickle.dump(
                {
                    "encodings": self.known_encodings,
                    "names": self.known_names
                },
                f
            )

    # Add a new person's face to the system by capturing 8 samples and averaging them
    def register_new_face(self, person_name, frames, num_samples=8):
        collected = 0
        temp_encodings = []

        print(f"📸 Registering {person_name}...")

        for frame in frames:
            # Convert frame to RGB and find faces in it
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, boxes)

            for enc in encodings:
                # Collect face encodings until we get enough samples
                temp_encodings.append(enc)
                collected += 1
                if collected >= num_samples:
                    break

            if collected >= num_samples:
                break

        if temp_encodings:
            # Average all the face samples to create one unique face representation
            avg_encoding = np.mean(temp_encodings, axis=0)
            self.known_encodings.append(avg_encoding)
            self.known_names.append(person_name)
            self.save_db()

            return True

        return False

    # Find who this face belongs to by comparing it to all saved faces
    def identify_face(self, encoding):

        if not self.known_encodings:
            return "Unknown"

        # Compare this face with all known faces to find closest match
        distances = face_recognition.face_distance(self.known_encodings, encoding)
        idx = np.argmin(distances)

        if distances[idx] <= 0.45:
            return self.known_names[idx]

        return "Unknown"


# Functions that are called from Module4

def register_face(person_name, frames):
    # Register a new person's face in the system

    recognizer = FaceRecognitionModule()

    success = recognizer.register_new_face(person_name, frames)

    if success:
        return f"Success"

    return "Face registration failed."


def recognize_face(frame):
    # Find out who is in front of the camera

    recognizer = FaceRecognitionModule()

    if frame is None:
        return "Camera capture failed."

    # Convert frame to RGB for face recognition
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    if not encodings:
        return "I do not see any face."

    # Check if we know who this person is
    name = recognizer.identify_face(encodings[0])
    if name == "Unknown":
        return "I do not recognize this person."

    return f"{name} is in front of you."