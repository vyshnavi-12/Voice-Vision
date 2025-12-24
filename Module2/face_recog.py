import cv2
import face_recognition
import pickle
import os
import numpy as np

DB_PATH = "face_db.pkl"


def load_database():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "rb") as f:
            return pickle.load(f)
    return {"encodings": [], "names": []}


def save_database(db):
    with open(DB_PATH, "wb") as f:
        pickle.dump(db, f)


def register_person(db, person_name, num_samples=8):
    cap = cv2.VideoCapture(0)
    print(f"[INFO] Registering {person_name}. Look at camera from different angles...")

    collected = 0
    temp_encodings = []  # Collect multiple per person
    
    while collected < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for (top, right, bottom, left), enc in zip(boxes, encodings):
            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Sample {collected+1}/{num_samples}",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            temp_encodings.append(enc)  # Store temporarily
            collected += 1
            if collected >= num_samples:
                break

        cv2.imshow("Register", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    
    # **NEW: Average encodings for better matching**
    if len(temp_encodings) > 0:
        avg_encoding = np.mean(temp_encodings, axis=0)
        db["encodings"].append(avg_encoding)
        db["names"].append(person_name)
        print(f"[INFO] Registered {person_name} with averaged encoding from {len(temp_encodings)} samples.")
    else:
        print("[WARN] No faces detected!")
    
    save_database(db)


def recognize_loop(db, tolerance=0.35):
    if len(db["encodings"]) == 0:
        print("[WARN] No known faces in database. Register someone first.")
        return

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting recognition. Press 'q' to quit.")

    known_encodings = db["encodings"]
    known_names = db["names"]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for (top, right, bottom, left), face_encoding in zip(boxes, encodings):
            # Compute distances to all known encodings
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(distances) == 0:
                name = "Unknown"
            else:
                best_idx = np.argmin(distances)
                best_distance = distances[best_idx]
                if best_distance <= tolerance:
                    name = known_names[best_idx]
                else:
                    name = "Unknown"

            # Draw results
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(
                frame,
                (left, bottom - 20),
                (right, bottom),
                (0, 0, 255),
                cv2.FILLED,
            )
            cv2.putText(
                frame,
                f"{name} ({best_distance:.2f})",
                (left + 2, bottom - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        cv2.imshow("Recognize", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    db = load_database()

    mode = input("Enter mode: [r]egister / [t]rack existing: ").strip().lower()
    if mode == "r":
        name = input("Enter person's name: ").strip()
        register_person(db, name, num_samples=8)
    else:
        recognize_loop(db, tolerance=0.45)

