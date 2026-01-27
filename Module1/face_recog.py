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

    def get_face_description(self, frame, face_location):
        top, right, bottom, left = face_location
        f_height, f_width, _ = frame.shape

        # -------- 1. CLOTHING COLOR (shirt region) --------
        shirt_top = min(bottom + 20, f_height - 1)
        shirt_bottom = min(bottom + 120, f_height)
        shirt_roi = frame[shirt_top:shirt_bottom, left:right]

        if shirt_roi.size > 0:
            avg_color = np.mean(shirt_roi.reshape(-1, 3), axis=0)  # BGR
            b, g, r = avg_color

            if r > 180 and g < 120:
                color_desc = "red"
            elif b > 180:
                color_desc = "blue"
            elif g > 160 and r < 120:
                color_desc = "green"
            elif r > 200 and g > 200 and b > 200:
                color_desc = "white"
            elif r < 60 and g < 60 and b < 60:
                color_desc = "dark or black"
            else:
                color_desc = "neutral-colored"
        else:
            color_desc = "unknown-colored"

        # -------- 2. RELATIVE HEIGHT --------
        face_center_y = (top + bottom) / 2
        if face_center_y < f_height * 0.35:
            height_desc = "taller than the camera's baseline"
        elif face_center_y > f_height * 0.6:
            height_desc = "shorter than the camera's baseline"
        else:
            height_desc = "about average height"

        # -------- 3. SPEAKING / MOTION (placeholder) --------
        motion_desc = "standing still"

        return f"wearing a {color_desc} top, appearing {height_desc}, and {motion_desc}"


    def run(self):
        last_seen_encoding=None  # seconds
        FACE_CHANGE_THRESHOLD=0.55

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
                desc = self.get_face_description(frame, loc)

                # -------- PRINT ONLY IF A DIFFERENT FACE APPEARS --------
                if last_seen_encoding is None:
                    print(f"🧍 Person in front is {desc}.")
                    last_seen_encoding = enc
                else:
                    diff = face_recognition.face_distance([last_seen_encoding], enc)[0]
                    if diff > FACE_CHANGE_THRESHOLD:
                        print(f"🧍 Person in front is {desc}.")
                        last_seen_encoding = enc


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
                    label = f"{name} | {desc}"

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
