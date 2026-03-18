import cv2
import os
from ultralytics import YOLO
import google.generativeai as genai
from dotenv import load_dotenv


# Load API keys and AI models
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

gemini_model = genai.GenerativeModel("gemini-2.5-flash")

# Load the YOLO object detection model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "yolov8x.pt")

yolo_model = YOLO(MODEL_PATH)


# Count how many people are visible in the camera frame
def count_people(frame):

    if frame is None:
        return -1

    results = yolo_model(frame, verbose=False)[0]
    people_count = 0

    for box in results.boxes:
        class_id = int(box.cls[0])
        # Class 0 in YOLO is person
        if class_id == 0:
            people_count += 1

    return people_count


# Describe what the person in front of the camera looks like
def describe_person(frame):

    if frame is None:
        return "Camera capture failed."

    try:
        # Convert frame to JPEG format
        success, encoded_image = cv2.imencode(".jpg", frame)
        if not success:
            return "I am having trouble seeing the image."

        image_part = {
            "mime_type": "image/jpeg",
            "data": encoded_image.tobytes()
        }

        # Ask Gemini AI to describe the person
        prompt = (
            "You are assisting a blind person. "
            "Describe the person in front of the camera in one short sentence. "
            "Mention clothing or appearance only briefly."
        )

        response = gemini_model.generate_content([prompt, image_part])
        return response.text.strip()

    except Exception as e:
        return f"AI Error: {str(e)}"