import cv2
import os
from ultralytics import YOLO
from Module1.object_classes import OBJECT_CLASSES

# standard path setup so the model loads regardless of where I run the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# using the 'x' (extra large) model for better accuracy in general detection
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8x.pt")

model = YOLO(MODEL_PATH)


# this function scans the room and lists what it sees
def detect_objects(frame):

    if frame is None:
        print("Frame error")
        return []

    # grabbing the first result from the model output
    results = model(frame, verbose=False)[0]

    detected_objects = []

    if results.boxes is None:
        return detected_objects

    for box in results.boxes:

        # get the name of the object (chair, person, etc.)
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        # ignore things that aren't in my pre-defined list (prevents noise)
        if label not in OBJECT_CLASSES:
            continue

        conf = float(box.conf[0])

        print(f"Detected: {label} ({conf:.2f})")

        detected_objects.append(label)

    # set() removes duplicates so I don't say 'chair' five times
    return list(set(detected_objects))