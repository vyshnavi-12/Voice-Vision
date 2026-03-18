import cv2
import os
from ultralytics import YOLO

# setting some constants for the math part later
KNOWN_HEIGHT = 1.0   # assuming the average obstacle is about 1 meter for math
FOCAL_LENGTH = 500   # this depends on the camera, need to calibrate if needed
SAFE_DISTANCE = 1.0  # alert the user if something is within 1 meter

# basic pathing setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# using my custom trained obstacle model
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "obstacle_best.pt")

model = YOLO(MODEL_PATH)

# fine-tuning the sensitivity
CONF_THRESHOLD = 0.65
MIN_BOX_HEIGHT = 100


# function to check if the user is about to walk into something
def detect_obstacle(frame):

    if frame is None:
        print("Frame error")
        return False

    # run the frame through the yolo model
    results = model(frame, verbose=False)[0]

    if results.boxes is None:
        return False

    for box in results.boxes:

        cls_id = int(box.cls[0])
        label = model.names[cls_id].lower()
        confidence = float(box.conf[0])

        # skip the weak detections to avoid false alarms
        if confidence < CONF_THRESHOLD:
            continue

        # getting the bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        # finding out how tall the object looks in pixels
        pixel_height = y2 - y1

        # if the box is tiny, it's probably too far away to care about
        if pixel_height < MIN_BOX_HEIGHT:
            continue

        # Simple triangle similarity formula for distance estimation (Monocular distance estimation)
        distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / pixel_height

        print(f"Detected: {label} ({confidence:.2f}) distance: {distance:.2f}m")

        # The 'Stop' trigger logic
        if distance <= SAFE_DISTANCE:
            print("⚠ Obstacle detected within safe distance")
            return True

    return False