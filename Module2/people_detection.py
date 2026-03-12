import cv2
import os
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv


# -------------------- SETUP --------------------

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("gemini-2.5-flash")

# Load YOLO once
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8x.pt")

model = YOLO(MODEL_PATH)


# -------------------- PEOPLE COUNT --------------------

def count_people():

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return -1   # camera error

    results = model(frame, verbose=False)[0]

    people_count = 0

    for box in results.boxes:
        class_id = int(box.cls[0])

        # COCO class 0 = person
        if class_id == 0:
            people_count += 1

    return people_count


# -------------------- PERSON DESCRIPTION --------------------

def describe_person():

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Camera capture failed."

    try:

        success, encoded_image = cv2.imencode(".jpg", frame)

        if not success:
            return "I am having trouble seeing the image."

        image_part = {
            "mime_type": "image/jpeg",
            "data": encoded_image.tobytes()
        }

        prompt = (
            "You are assisting a blind person. "
            "Describe the person in front of the camera in one short sentence. "
            "Mention clothing or appearance only briefly."
        )

        response = model.generate_content([prompt, image_part])

        return response.text.strip()

    except Exception as e:
        return f"AI Error: {str(e)}"