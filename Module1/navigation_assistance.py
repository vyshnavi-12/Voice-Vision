import cv2
import os
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv


# -------------------------------
# Load Gemini API Key
# -------------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
    sys.exit()

# -------------------------------
# Initialize Gemini Client
# -------------------------------
client = genai.Client(api_key=api_key)


# -------------------------------
# AI Object Navigation
# -------------------------------
def find_object(frame, target_object):

    try:
        print(f"[Vision] Searching for '{target_object}'")

        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            return "Camera capture error."

        image_part = types.Part.from_bytes(
            data=buffer.tobytes(),
            mime_type="image/jpeg"
        )

        prompt = (
            f"You are a guide assistant for a blind person. "
            f"Search for the '{target_object}'. "
            "If found: give direction using clock face "
            "(12 straight, 3 right, 9 left) and estimate distance. "
            "If not found say: 'I do not see the object'. "
            "Keep response extremely short."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_part]
        )

        return response.text.strip()

    except Exception as e:
        return f"Navigation AI Error: {e}"


# -------------------------------
# Capture Frame + Navigate
# -------------------------------
def navigate_to_object(target_object):

    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    cap.release()

    if not ret:
        return "Camera capture failed."

    guidance = find_object(frame, target_object)

    return guidance