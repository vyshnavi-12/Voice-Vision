import cv2
import os
from ultralytics import YOLO

# -------------------------------
# Distance Parameters
# -------------------------------
KNOWN_HEIGHT = 1.0
FOCAL_LENGTH = 500
SAFE_DISTANCE = 1.0

# -------------------------------
# Load YOLO Model
# -------------------------------
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "obstacle_best.pt")

model = YOLO(MODEL_PATH)

# -------------------------------
# Detection Parameters
# -------------------------------
CONF_THRESHOLD = 0.65
MIN_BOX_HEIGHT = 100


# -------------------------------
# Detect obstacle from frame
# -------------------------------
def detect_obstacle_from_frame(frame):

    results = model(frame, verbose=False)[0]

    if results.boxes is None:
        return False

    for box in results.boxes:

        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower()
        confidence = float(box.conf[0])

        if confidence < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        pixel_height = y2 - y1

        # ignore very small detections
        if pixel_height < MIN_BOX_HEIGHT:
            continue

        distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / pixel_height

        print(f"Detected: {label} ({confidence:.2f}) distance: {distance:.2f}m")

        if distance <= SAFE_DISTANCE:
            print("⚠ Obstacle detected within safe distance")
            return True

    return False


# -------------------------------
# Capture frame from webcam
# -------------------------------
def detect_obstacle():

    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    cap.release()

    if not ret:
        print("Camera error")
        return False

    obstacle = detect_obstacle_from_frame(frame)

    return obstacle