import cv2
import os
from ultralytics import YOLO

# getting the paths right so it doesn't crash on different folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# pointing to where I saved the currency model weights
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "currency_best.pt")

# load the YOLO model once here so we don't lag the camera later
model = YOLO(MODEL_PATH)

# main function to process the frame and find money
def detect_currency(frame):

    if frame is None:
        return None

    # low conf here just to get initial results, will filter strictly below
    results = model(frame, conf=0.6, verbose=False)

    detected_currency = None

    for r in results:

        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:

            # Only pick it up if the model is 70% sure, otherwise ignore it
            conf = float(box.conf[0])
            if conf < 0.70:
                continue

            # grab the name (like '100 Rupees') using the class ID
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            detected_currency = label
            break

    return detected_currency