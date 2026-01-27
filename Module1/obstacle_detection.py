import cv2
import vision_utils
import distance_estimation  # <--- Importing the math from File 2

SAFE_DISTANCE = 1.5 # meters

def run():
    model = vision_utils.load_model()
    cap = cv2.VideoCapture(0)
    
    print(f"Obstacle Mode Active. Stop Limit: {SAFE_DISTANCE}m")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # conf=0.25 ensures we catch the bottle even if Person is dominant
        results = model.predict(frame, conf=0.25, iou=0.5, verbose=False)
        
        for result in results:
            for box in result.boxes:
                # 1. Get Name & Coords
                cls_id = int(box.cls[0])
                name = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 2. Get Distance from the Distance Module
                dist = distance_estimation.get_distance(box)

                # --- 3. SAFETY LOGIC ---
                
                # CHECK: Is it in the HAZARD list? (e.g., Chair, Person)
                if name in vision_utils.HAZARDS:
                    if dist < SAFE_DISTANCE:
                        # DANGER
                        color = (0, 0, 255) # Red
                        text = f"STOP! {name} {dist:.1f}m"
                    else:
                        # WARNING
                        color = (0, 255, 255) # Yellow
                        text = f"Warn: {name} {dist:.1f}m"
                
                # CHECK: Is it a SAFE ITEM? (e.g., Book, Bottle)
                else:
                    color = (0, 255, 0) # Green
                    text = f"Safe: {name} {dist:.1f}m"

                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Mode: Obstacle Safety', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()