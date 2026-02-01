import cv2                          # OpenCV for camera and image processing
import mediapipe as mp              # MediaPipe for face detection (people count)
import google.generativeai as genai  # Gemini API for natural description
import threading                    # Run Gemini calls without freezing camera
import queue                        # Safe communication between threads
import sys                          # Read terminal input
import os                           # Access environment variables
from dotenv import load_dotenv      # Load API key securely

# --- SETUP ---
load_dotenv()  # Load .env file
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))  # Configure Gemini
model = genai.GenerativeModel('gemini-2.5-flash')    # Fast multimodal model

class PersonDescriptionModule:
    def __init__(self):
        # Initialize MediaPipe Face Detection (accurate for counting people)
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,          # Best for close-range camera usage
            min_detection_confidence=0.7
        )
        print("\n[SYSTEM] Voice Vision: MediaPipe + Gemini Hybrid Active.")

    def count_people(self, frame):
        """Detects faces and returns number of people."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)

        count = 0
        if results.detections:
            count = len(results.detections)
            # Draw face boxes for visual feedback (optional)
            for detection in results.detections:
                mp.solutions.drawing_utils.draw_detection(frame, detection)

        return f"There are {count} people in front of you."

    def get_natural_description(self, frame):
        """Uses Gemini Vision to describe the person naturally."""
        try:
            # Convert frame to JPEG for Gemini input
            success, encoded_image = cv2.imencode('.jpg', frame)
            if not success:
                return "I'm having trouble seeing the image."

            image_part = {
                "mime_type": "image/jpeg",
                "data": encoded_image.tobytes()
            }

            # Prompt tuned for blind-user friendly narration
            prompt = (
                "You are an assistant for a blind person. Briefly describe the person "
                "standing directly in front of the camera. Mention their gender, "
                "what they are wearing, and any notable accessories. "
                "Speak naturally."
            )

            response = model.generate_content([prompt, image_part])
            return response.text.strip()

        except Exception as e:
            return f"Sorry, I hit an error: {str(e)}"

# --- TERMINAL INPUT THREAD ---
def terminal_listener(q):
    # Continuously listens for user commands from terminal
    while True:
        line = sys.stdin.readline().strip().lower()
        if line:
            q.put(line)

# --- MAIN PROGRAM ---
if __name__ == "__main__":
    describer = PersonDescriptionModule()
    cap = cv2.VideoCapture(0)  # Open webcam

    cmd_queue = queue.Queue()
    # Run terminal input listener in background
    threading.Thread(target=terminal_listener, args=(cmd_queue,), daemon=True).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Show live camera feed
        cv2.imshow("Voice Vision Feed", cv2.flip(frame, 1))
        cv2.waitKey(1)

        if not cmd_queue.empty():
            cmd = cmd_queue.get()

            if cmd == 'exit':
                break

            elif cmd == 'count':
                print(f"VOICE VISION: {describer.count_people(frame)}")

            elif cmd == 'describe':
                print("[AI] Looking at the person...")

                # Run Gemini call in a separate thread
                def api_thread():
                    msg = describer.get_natural_description(frame)
                    print(f"VOICE VISION: {msg}")

                threading.Thread(target=api_thread).start()

    cap.release()
    cv2.destroyAllWindows()
