from pyexpat import model
import cv2
from ultralytics import YOLO  # Import the YOLO AI model

def start_detection():
    # 1. Load the AI Model
    # 'yolov8n.pt' is the "Nano" version. It is the fastest and best for laptops.
    # On the first run, it will automatically download this file (about 6MB).
    print("Loading AI Model...")
    model = YOLO('yolov8m.pt') 

    # 2. Start Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    print("System Active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 3. Run Object Detection
        # We pass the current 'frame' to the model.
        # conf=0.65 means "Don't tell me unless you are 65% sure"
        # iou=0.45 helps remove duplicate boxes for the same object
        results = model(frame, conf=0.65, iou=0.45)

        # 4. Draw the Results
        # 'plot()' automatically draws the bounding boxes and labels (like 'person', 'cup')
        # on the frame for us.
        annotated_frame = results[0].plot()

        # 5. Display the Output
        cv2.imshow('Voice Vision - Phase 2 (Object Detection)', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_detection()