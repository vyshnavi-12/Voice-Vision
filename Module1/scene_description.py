import cv2                         # Camera access and image handling
import os                          
import sys                         
from google import genai           # Modern Gemini client
from google.genai import types     # Required for structured image input
from dotenv import load_dotenv     


# 1. ENVIRONMENT & API SETUP
# Loads environment variables and safely retrieves
# the Gemini API key needed for vision + language tasks.
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Stop execution if API key is missing
if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
    sys.exit()

# 2. GEMINI CLIENT INITIALIZATION
# Creates a Gemini client using the new SDK that
# supports multimodal (image + text) requests.
client = genai.Client(api_key=api_key)


def analyze_scene(frame):
    """
    Sends a single camera frame to Gemini and returns
    a short, blind-friendly description of the scene.
    """
    try:
        print("\n[System] Analyzing scene...")

        # IMAGE ENCODING
        # Converts the OpenCV frame into JPEG bytes
        # so it can be sent to Gemini.
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            return "Capture error."

        # IMAGE WRAPPING FOR GEMINI
        # Gemini expects images to be wrapped in a
        # structured Part with MIME type information.
        image_part = types.Part.from_bytes(
            data=buffer.tobytes(),
            mime_type="image/jpeg"
        )

        # PROMPT DESIGN (SCENE DESCRIPTION)
        # Instructs the AI to:
        # - Focus on critical foreground information first
        # - Keep output short and clear
        # - Be suitable for blind navigation
        prompt = (
            "You are a mobility assistant for a blind person. "
            "Describe the scene briefly. Start with the most important thing "
            "(like a person or obstacle) right in front of the camera, "
            "then mention the background. Keep it to 2 short sentences."
        )

        # MULTIMODAL REQUEST
        # Sends both the text prompt and the image
        # to Gemini and retrieves the description.
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_part]
        )

        return response.text.strip()

    except Exception as e:
        # Handles encoding, API, or runtime failures gracefully
        return f"AI Error: {e}"

# 3. APPLICATION ENTRY POINT
# Opens the camera, displays live feed, and allows
# the user to trigger scene description using a key.
if __name__ == "__main__":

    cap = cv2.VideoCapture(0)       # Open default webcam

    # User guidance for interaction
    print("\n" + "="*40)
    print(" VOICE VISION: SCENE DESCRIPTION ")
    print(" 1. CLICK THE CAMERA WINDOW FIRST")
    print(" 2. Press SPACE to Describe")
    print(" 3. Press 'Q' to Quit")
    print("="*40 + "\n")


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # LIVE CAMERA FEED
        # Flipped horizontally for natural, mirror-like view.
        display_frame = cv2.flip(frame, 1)
        cv2.imshow("Voice Vision Feed", display_frame)

        key = cv2.waitKey(30) & 0xFF

        # SCENE DESCRIPTION TRIGGER
        # Pressing SPACE captures the current frame
        # and sends it to Gemini for analysis.
        if key == 32:
            description = analyze_scene(frame)
            print(f"\n[AI RESULT]: {description}")
            print("-" * 40)

        # EXIT CONDITION
        # Cleanly closes the application.
        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
