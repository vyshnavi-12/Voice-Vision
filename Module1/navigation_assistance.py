import google.generativeai as genai
import cv2
from PIL import Image

# --- CONFIGURATION ---
API_KEY = "AIzaSyAX3COkVgNTlulTlCaJqY5rBSZmP0zvYy8"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

def find_object(frame, target_object):
    """
    Asks Gemini to find a specific object in the frame and 
    give directional guidance (Clock Face).
    """
    try:
        print(f"\n [System] Looking for: '{target_object}'...")
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        # THE NAVIGATION PROMPT
        # We force the AI to act like a Guide Dog
        prompt = (
            f"I am a visually impaired person. I need to find the '{target_object}'. "
            "Look at this image. "
            "1. If you see it, tell me its location using 'Clock Face' directions "
            "(e.g., 'at 12 o'clock' for straight, '3 o'clock' for right). "
            "2. Estimate the distance in steps or meters. "
            "3. If it is NOT there, simply say 'I do not see the {target_object}'."
            "Keep the answer short and direct."
        )

        response = model.generate_content([prompt, pil_image])
        return response.text.strip()

    except Exception as e:
        return f"Error: {e}"

# --- TEST LOOP ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    print("------------------------------------------------")
    print(" NAVIGATION MODULE (CONSOLE MODE)               ")
    print(" 1. Press 'N' to search for an object.          ")
    print(" 2. Press 'Q' to quit.                          ")
    print("------------------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret: break

        cv2.imshow("Navigation Assistant", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('n'):
            # 1. Ask user what to find
            print("\n------------------------------------------------")
            target = input(" [INPUT] What do you want to find? (e.g. door): ")
            
            # 2. Find it
            if target.strip():
                guidance = find_object(frame, target)
                print(f" [AI Guide]: {guidance}")
                print("------------------------------------------------")
        
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()