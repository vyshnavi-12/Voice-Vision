import cv2
import os
import sys
from google import genai
from google.genai import types  # Helper for strictly formatted data
from dotenv import load_dotenv

# 1. ENVIRONMENT SETUP
# Securely load the API key from your .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("Error: GEMINI_API_KEY not found in .env file.")
    sys.exit()

# 2. CLIENT INITIALIZATION
# Using the 2026 'google-genai' SDK for multimodal support
client = genai.Client(api_key=api_key)

def find_object(frame, target_object):
    """
    Identifies target object and provides Clock Face directional guidance.
    """
    try:
        print(f"\n[System] Locating '{target_object}'...")
        
        # Encode image to JPEG bytes for the API
        success, buffer = cv2.imencode('.jpg', frame)
        if not success: return "Capture error."

        # WRAP IMAGE DATA: Ensuring the AI receives the correct 'Part' structure
        image_part = types.Part.from_bytes(
            data=buffer.tobytes(),
            mime_type="image/jpeg"
        )

        # UPDATED PROMPT: Force-aligned for blind navigation safety
        prompt = (
            f"You are a guide assistant for a blind person. Search for the '{target_object}'. "
            "If found: 1. Give direction using 'Clock Face' (12=straight, 3=right, 9=left). "
            "2. Estimate distance in steps or meters. 3. If NOT found, say: 'I do not see the {target_object}'. "
            "Keep instructions extremely brief and clear."
        )

        # Multimodal request combining text and the image part
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=[prompt, image_part]
        )
        return response.text.strip()

    except Exception as e:
        return f"Navigation AI Error: {e}"

# 3. INTERACTIVE NAVIGATION LOOP
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    print("-" * 48)
    print(" VOICE VISION: NAVIGATION ASSISTANT             ")
    print(" - Press 'N' to search for an object.           ")
    print(" - Press 'Q' to quit.                           ")
    print("-" * 48)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Show live monitor (flipped for natural mirror movement)
        cv2.imshow("Navigation Monitor", cv2.flip(frame, 1))
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n') or key == ord('N'):
            # Allow user to type the target in the console
            print("-" * 48)
            target = input(" [INPUT] What should I look for? (e.g., cup): ")
            
            if target.strip():
                # Execute AI search
                guidance = find_object(frame, target)
                print(f" VOICE VISION: {guidance}")
                print("-" * 48)
        
        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()