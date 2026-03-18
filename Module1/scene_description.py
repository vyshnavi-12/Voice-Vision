import cv2
import os
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv


# loading the api key from the env file for security
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
    sys.exit()

# setting up the connection to the ai model
client = genai.Client(api_key=api_key)


# this function gets a summary of what's happening in front of the user
def analyze_scene(frame):

    try:
        print("\n[Vision] Sending frame to Gemini...")

        # turning the camera frame into a jpg so it can be sent over the network
        success, buffer = cv2.imencode('.jpg', frame)

        if not success:
            return "Image capture error."

        # converting the image bytes into a format the gemini api understands
        image_part = types.Part.from_bytes(
            data=buffer.tobytes(),
            mime_type="image/jpeg"
        )

        # telling the ai exactly how to speak to the user—brief and helpful
        prompt = (
            "You are a mobility assistant for a blind person. "
            "Describe the scene briefly. Start with the most important thing "
            "(like a person or obstacle) right in front of the camera, "
            "then mention the background. Keep it to 2 short sentences."
        )

        # using the 2.5-flash model here for a fast response
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_part]
        )

        return response.text.strip()

    except Exception as e:
        return f"AI Error: {e}"


# simple wrapper to handle the frame from the frontend
def describe_scene(frame):

    if frame is None:
        return "Camera capture failed."

    description = analyze_scene(frame)

    return description