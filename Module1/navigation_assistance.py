import cv2  # camera handling
import os
import sys
from google import genai
from google.genai import types
from dotenv import load_dotenv

# 1. ENVIRONMENT & API SETUP
# Loads environment variables and safely fetches
# the Gemini API key required for multimodal requests.
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
    sys.exit()

# 2. GEMINI CLIENT INITIALIZATION
# Initializes the Google GenAI client using the
# latest SDK that supports image + text inputs.
client = genai.Client(api_key=api_key)

def find_object(frame, target_object):
    """
    Uses a single camera frame to locate a target object
    and returns short, clock-face–based navigation guidance
    suitable for blind users.
    """
    try:
        print(f"\n[System] Locating '{target_object}'...")

        # IMAGE PREPARATION
        # Convert the OpenCV frame into JPEG bytes so it
        # can be sent to Gemini as a multimodal input.
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            return "Capture error."

        # IMAGE WRAPPING FOR GEMINI
        # Gemini requires image data to be wrapped
        # as a structured Part with MIME type info.
        image_part = types.Part.from_bytes(
            data=buffer.tobytes(),
            mime_type="image/jpeg"
        )

        # PROMPT ENGINEERING (NAVIGATION-SAFE)
        # Forces short, structured output using
        # clock-face directions and distance estimation.
        prompt = (
            f"You are a guide assistant for a blind person. Search for the '{target_object}'. "
            "If found: 1. Give direction using 'Clock Face' (12=straight, 3=right, 9=left). "
            "2. Estimate distance in steps or meters. "
            "3. If NOT found, say: 'I do not see the {target_object}'. "
            "Keep instructions extremely brief and clear."
        )

        # MULTIMODAL AI REQUEST
        # Sends both the prompt and image to Gemini
        # and retrieves the generated navigation guidance.
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_part]
        )

        return response.text.strip()

    except Exception as e:
        return f"Navigation AI Error: {e}"

# 3. INTERACTIVE NAVIGATION LOOP
# Continuously captures camera frames, listens for
# user key input, and performs object search on demand.
if __name__ == "__main__":

    # Open default camera
    cap = cv2.VideoCapture(0)

    print("-" * 48)
    print(" VOICE VISION: NAVIGATION ASSISTANT             ")
    print(" - Press 'N' to search for an object.           ")
    print(" - Press 'Q' to quit.                           ")
    print("-" * 48)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # LIVE CAMERA PREVIEW
        # Flipped horizontally for natural mirror-like view.
        cv2.imshow("Navigation Monitor", cv2.flip(frame, 1))
        key = cv2.waitKey(1) & 0xFF

        # OBJECT SEARCH TRIGGER
        # Captures the current frame and sends it to Gemini
        # when the user presses 'N'.
        if key == ord('n') or key == ord('N'):
            print("-" * 48)
            target = input(" [INPUT] What should I look for? (e.g., cup): ")

            if target.strip():
                guidance = find_object(frame, target)
                print(f" VOICE VISION: {guidance}")
                print("-" * 48)

        # EXIT CONDITION
        # Cleanly shuts down camera and UI.
        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
