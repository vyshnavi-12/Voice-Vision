import cv2                         # Camera access and image handling
import os                          # Environment variable access
import sys                         # Program exit handling
from google import genai           # Modern Gemini client
from google.genai import types     # Required for structured image input
from dotenv import load_dotenv     # Load API key from .env file

# 1. SETUP
load_dotenv()                      # Load environment variables
api_key = os.getenv("GEMINI_API_KEY")

# Ensure API key exists before continuing
if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
    sys.exit()

# Initialize Gemini Client (new SDK style)
client = genai.Client(api_key=api_key)

def analyze_scene(frame):
    """
    Captures a camera frame and sends it to Gemini
    for a natural scene description.
    """
    try:
        print("\n[System] Analyzing scene...")

        # Encode OpenCV frame to JPEG bytes
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            return "Capture error."

        # Wrap image bytes using Gemini's Part structure
        image_part = types.Part.from_bytes(
            data=buffer.tobytes(),
            mime_type="image/jpeg"
        )

        # Simple, blind-friendly prompt
        prompt = (
            "You are a mobility assistant for a blind person. "
            "Describe the scene briefly. Start with the most important thing "
            "(like a person or obstacle) right in front of the camera, "
            "then mention the background. Keep it to 2 short sentences."
        )

        # Send prompt + image to Gemini
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_part]
        )

        return response.text.strip()

    except Exception as e:
        # Handles API, encoding, or runtime errors
        return f"AI Error: {e}"

# --- APPLICATION ENTRY POINT ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)       # Open webcam

    # User instructions
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

        # Mirror the feed for natural interaction
        display_frame = cv2.flip(frame, 1)
        cv2.imshow("Voice Vision Feed", display_frame)

        key = cv2.waitKey(30) & 0xFF

        # SPACE key triggers scene description
        if key == 32:
            description = analyze_scene(frame)
            print(f"\n[AI RESULT]: {description}")
            print("-" * 40)

        # Quit application
        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()
