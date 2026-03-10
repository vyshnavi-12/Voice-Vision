import cv2
import os
from ultralytics import YOLO

# -----------------------------
# LOAD MODEL ONCE
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "currency_best.pt")

model = YOLO(MODEL_PATH)
# -----------------------------
# DETECT CURRENCY FROM FRAME
# -----------------------------
def detect_currency():

    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    cap.release()

    if not ret:
        return None

    results = model(frame, conf=0.6, verbose=False)

    detected_currency = None

    for r in results:

        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:

            # -------- Confidence Filter --------
            conf = float(box.conf[0])
            if conf < 0.70:
                continue
            # -----------------------------------

            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            detected_currency = label
            break

    return detected_currency