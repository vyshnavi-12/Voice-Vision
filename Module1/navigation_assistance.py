import cv2
import os
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv


# getting the api key from the .env file so it's secure
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
    sys.exit()

# connecting to the google ai client
client = genai.Client(api_key=api_key)


# this function uses the gemini model to 'see' and find stuff
def find_object(frame, target_object):

    try:
        print(f"[Vision] Searching for '{target_object}'")

        # convert the camera frame to a jpg so the api can read it
        success, buffer = cv2.imencode(".jpg", frame)

        if not success:
            return "Camera capture error."

        # wrap the image data in a format the gemini api understands
        image_part = types.Part.from_bytes(
            data=buffer.tobytes(),
            mime_type="image/jpeg"
        )

        # setting the persona so the ai knows it's helping a blind person
        prompt = (
            f"You are a guide assistant for a blind person. "
            f"Search for the '{target_object}'. "
            "If found: give direction using clock face "
            "(12 straight, 3 right, 9 left) and estimate distance. "
            "If not found say: 'I do not see the object'. "
            "Keep response extremely short."
        )

        # hitting the flash model (it's the fastest one for live video)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_part]
        )

        return response.text.strip()

    except Exception as e:
        return f"Navigation AI Error: {e}"


# helper function to just pass the frame along
def navigate_to_object(frame, target_object):

    if frame is None:
        return "Camera capture failed."

    guidance = find_object(frame, target_object)

    return guidance