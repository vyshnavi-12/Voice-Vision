import cv2                          # OpenCV for camera access and image processing
import google.generativeai as genai  # Gemini API for natural image description
import threading                    # Run Gemini calls without blocking camera feed
import queue                        # Thread-safe communication between threads
import sys                          # Read terminal input commands
import os                           # Access environment variables
from dotenv import load_dotenv      # Load API keys securely from .env file
from ultralytics import YOLO        # YOLOv8 for accurate people detection

# -------------------- SETUP --------------------

# Load environment variables (GEMINI_API_KEY)
load_dotenv()

# Configure Gemini with API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Fast multimodal Gemini model (image + text)
model = genai.GenerativeModel('gemini-2.5-flash')


class PersonDescriptionModule:
    def __init__(self):
        """
        Initialize AI models used in the system.
        """

        # Load YOLOv8 model (lightweight & fast)
        # This model is trained on the COCO dataset
        # Class ID 0 corresponds to 'person'
        self.yolo = YOLO("yolov8n.pt")

        print("\n[SYSTEM] Voice Vision: YOLO + Gemini Active.")

    def count_people(self, frame):
        """
        Detects and counts people using YOLO.

        Why YOLO?
        - Detects full human bodies (not just faces)
        - Works well for groups and side angles
        - More accurate than face-based counting
        """

        # Run YOLO inference on the current frame
        results = self.yolo(frame, verbose=False)[0]

        people_count = 0

        # Iterate through all detected objects
        for box in results.boxes:
            class_id = int(box.cls[0])

            # COCO class 0 = person
            if class_id == 0:
                people_count += 1

                # Draw bounding box around detected person
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

        return f"There are {people_count} people in front of you."

    def get_natural_description(self, frame):
        """
        Uses Gemini Vision to generate a natural-language
        description of the person in front of the camera.
        """

        try:
            # Convert OpenCV frame to JPEG format
            success, encoded_image = cv2.imencode('.jpg', frame)
            if not success:
                return "I'm having trouble seeing the image."

            # Prepare image data for Gemini
            image_part = {
                "mime_type": "image/jpeg",
                "data": encoded_image.tobytes()
            }

            # Prompt designed for blind-friendly narration
            prompt = (
                "You are an assistant for a blind person. "
                "Briefly describe the person standing directly in front of the camera. "
                "Mention their clothing and any notable accessories. "
                "Speak naturally."
            )

            # Send prompt + image to Gemini
            response = model.generate_content([prompt, image_part])

            return response.text.strip()

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"


# ---------------- TERMINAL INPUT THREAD ----------------

def terminal_listener(cmd_queue):
    """
    Continuously listens for user commands from terminal.
    Runs in a background thread so camera feed is not blocked.
    """
    while True:
        command = sys.stdin.readline().strip().lower()
        if command:
            cmd_queue.put(command)


# -------------------- MAIN PROGRAM --------------------

if __name__ == "__main__":

    describer = PersonDescriptionModule()

    # Open default webcam
    cap = cv2.VideoCapture(0)

    # Queue for handling terminal commands safely
    cmd_queue = queue.Queue()

    # Start terminal listener in background thread
    threading.Thread(
        target=terminal_listener,
        args=(cmd_queue,),
        daemon=True
    ).start()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Show live camera feed (mirror view)
        cv2.imshow("Voice Vision Feed", cv2.flip(frame, 1))
        cv2.waitKey(1)

        # Process user commands
        if not cmd_queue.empty():
            cmd = cmd_queue.get()

            if cmd == "exit":
                break

            elif cmd == "count":
                print(f"VOICE VISION: {describer.count_people(frame)}")

            elif cmd == "describe":
                print("[AI] Looking at the person...")

                # Run Gemini description in separate thread
                def api_thread():
                    message = describer.get_natural_description(frame)
                    print(f"VOICE VISION: {message}")

                threading.Thread(target=api_thread).start()

    # Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
