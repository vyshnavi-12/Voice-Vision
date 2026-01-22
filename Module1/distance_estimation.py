import cv2
import math
from ultralytics import YOLO

def start_distance_estimation():
    # 1. Load the Smarter Model (Medium)
    # This reduces errors like confusing a bed stand for a bottle.
    print("Loading AI Model (Medium)...")
    model = YOLO('yolov8m.pt') 

    # 2. Setup Webcam
    cap = cv2.VideoCapture(0)
    
    # --- CALIBRATION VARIABLES ---
    # These numbers help us convert pixels to meters. 
    # Real width of a common object (e.g., an average person's shoulder width is ~0.5 meters)
    KNOWN_WIDTH = 0.5  
    # Focal length is camera-specific. 
    # 600 is a good average guess for laptop webcams. 
    # If the distance is wrong, increase this number (e.g. to 800) or decrease it.
    FOCAL_LENGTH = 600 
    # -----------------------------

    print("System Active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3. Run Detection
        results = model(frame, conf=0.5)
        
        # 4. Process Results
        # We need to loop through every object detected in the frame
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get the coordinates of the box: x1, y1, x2, y2
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(y2), int(y2)
                
                # Get the class name (e.g., 'person', 'chair')
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # --- DISTANCE CALCULATION LOGIC ---
                # Calculate the width of the object in pixels on your screen
                w_pixels = x2 - x1
                
                # Formula: Distance = (Real_Width * Focal_Length) / Pixel_Width
                # We protect against division by zero with max(1, w_pixels)
                distance = (KNOWN_WIDTH * FOCAL_LENGTH) / max(1, w_pixels)
                # ----------------------------------

                # 5. Draw the Box and Distance
                # Draw the rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Create the label text: "Person: 1.2m"
                label = f"{class_name}: {distance:.2f}m"
                
                # Put the text above the box
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 6. Display Output
        cv2.imshow('Voice Vision - Distance Estimation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_distance_estimation()