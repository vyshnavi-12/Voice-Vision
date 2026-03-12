import cv2
import os
from ultralytics import YOLO
from Module1.object_classes import OBJECT_CLASSES

# -------------------------------
# Load YOLO Model (only once)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8x.pt")

model = YOLO(MODEL_PATH)


# -------------------------------
# Detect objects from frame
# -------------------------------
def detect_objects_from_frame(frame):

    results = model(frame, verbose=False)[0]

    detected_objects = []

    if results.boxes is None:
        return detected_objects

    for box in results.boxes:

        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label not in OBJECT_CLASSES:
            continue

        conf = float(box.conf[0])

        print(f"Detected: {label} ({conf:.2f})")

        detected_objects.append(label)

    return list(set(detected_objects))


# -------------------------------
# Capture frame and detect
# -------------------------------
def detect_objects():

    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Camera error")
        return []

    objects = detect_objects_from_frame(frame)

    return objects