import cv2
import vision_utils

# --- CONFIGURATION ---
FOCAL_LENGTH = 600
KNOWN_WIDTH = 0.5  # meters

def get_distance(box):
    """
    This function calculates distance. 
    Other files (like obstacle_detection) will call THIS function.
    """
    # 1. Get Width in Pixels
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    width_px = x2 - x1
    
    # 2. The Math Formula
    # Distance = (Real_Width * Focal_Length) / Pixel_Width
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / max(1, width_px)
    return distance

def run():
    model = vision_utils.load_model()
    cap = cv2.VideoCapture(0)
    
    print("Distance Estimation Mode Active...")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # LOWER CONFIDENCE (0.2) helps detect small objects held in hand
        results = model.predict(frame, conf=0.2, iou=0.5, verbose=False)
        
        for result in results:
            for box in result.boxes:
                # 1. Ask THIS file to calculate distance
                dist = get_distance(box)
                
                # 2. Get Name
                cls_id = int(box.cls[0])
                name = model.names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # 3. Draw Blue Box (Info Only)
                label = f"{name}: {dist:.2f}m"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 100, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2)

        cv2.imshow('Mode: Distance Estimation', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()