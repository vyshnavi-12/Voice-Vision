import google.generativeai as genai
import cv2
from PIL import Image

# --- CONFIGURATION ---
# 1. Paste your API Key here
API_KEY = "AIzaSyAX3COkVgNTlulTlCaJqY5rBSZmP0zvYy8"

# 2. Configure Gemini
genai.configure(api_key=API_KEY)

# 3. Initialize Model (Using the fast 2.5 Flash)
model = genai.GenerativeModel('gemini-2.5-flash')

# --- CORE FUNCTION (To be imported later) ---
def analyze_scene(frame):
    """
    Takes a single video frame, sends it to Gemini, 
    and returns a text description.
    """
    try:
        print("\n [System] Sending image to AI...")
        
        # Convert OpenCV Frame (BGR) to PIL Image (RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # The Prompt
        prompt = (
            "Describe this scene for a blind person in 1 or 2 sentences. "
            "Focus on the general layout, major obstacles, and people."
        )

        # Generate Content
        response = model.generate_content([prompt, pil_image])
        
        # Return clean text
        return response.text.strip()

    except Exception as e:
        return f"Error: {e}"

# --- TEST LOOP (Runs only when you execute this file) ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    print("------------------------------------------------")
    print(" SCENE DESCRIPTION MODULE (CONSOLE MODE)        ")
    print(" 1. Press SPACEBAR to capture and describe.     ")
    print(" 2. Press 'Q' to quit.                          ")
    print("------------------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Show the video feed
        cv2.imshow("Scene Description (Press Space)", frame)
        
        # Check for Key Press
        key = cv2.waitKey(1) & 0xFF

        if key == 32: # 32 is the ASCII code for SPACEBAR
            
            # 1. Call the function
            description = analyze_scene(frame)
            
            # 2. Print result to Console
            print(f" [AI Result]: {description}")
            print("------------------------------------------------")
        
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()