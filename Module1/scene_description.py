import cv2
import os
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv


# -------------------------------
# Load API Key
# -------------------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
    sys.exit()

# -------------------------------
# Gemini Client
# -------------------------------
client = genai.Client(api_key=api_key)


# -------------------------------
# Scene Analysis Function
# -------------------------------
def analyze_scene(frame):

    try:
        print("\n[Vision] Sending frame to Gemini...")

        success, buffer = cv2.imencode('.jpg', frame)

        if not success:
            return "Image capture error."

        image_part = types.Part.from_bytes(
            data=buffer.tobytes(),
            mime_type="image/jpeg"
        )

        prompt = (
            "You are a mobility assistant for a blind person. "
            "Describe the scene briefly. Start with the most important thing "
            "(like a person or obstacle) right in front of the camera, "
            "then mention the background. Keep it to 2 short sentences."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_part]
        )

        return response.text.strip()

    except Exception as e:
        return f"AI Error: {e}"


# -------------------------------
# Capture Frame + Analyze
# -------------------------------
def describe_scene():

    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()

    cap.release()

    if not ret:
        return "Camera capture failed."

    description = analyze_scene(frame)

    return description