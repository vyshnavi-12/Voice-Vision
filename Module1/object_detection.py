import cv2
import vision_utils
from collections import deque, Counter

# --- STABILIZER SETTINGS ---
# How many frames to remember? (Higher = More stable, but slightly slower update)
HISTORY_LENGTH = 15 

# Dictionary to store the history of every object found
# Format: { object_id: ['bottle', 'bottle', 'remote', 'bottle'] }
object_history = {}

def get_stable_name(track_id, current_name):
    """
    Takes the new name and votes with the last 15 frames 
    to decide the TRUE name.
    """
    if track_id not in object_history:
        object_history[track_id] = deque(maxlen=HISTORY_LENGTH)
    
    # Add the new guess to history
    object_history[track_id].append(current_name)
    
    # Count the votes (e.g., {'bottle': 14, 'remote': 1})
    votes = Counter(object_history[track_id])
    
    # Return the winner (The most common name)
    most_common_name = votes.most_common(1)[0][0]
    return most_common_name

def run_stable_detection():
    model = vision_utils.load_model()
    cap = cv2.VideoCapture(0)

    # Set Window Size
    cv2.namedWindow('Voice Vision - Stabilized Mode', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Voice Vision - Stabilized Mode', 1000, 800)

    print("System Active with STABILIZER. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # --- KEY CHANGE: Use .track() instead of .predict() ---
        # persist=True tells the AI: "This is a video, remember objects from last frame"
        # tracker="bytetrack.yaml" is a standard built-in tracker configuration
        results = model.track(frame, persist=True, conf=0.25, iou=0.5, verbose=False)
        
        annotated_frame = frame.copy()
        
        # Check if we found anything
        if results[0].boxes.id is not None:
            
            # Get the boxes, class IDs, and Tracking IDs
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, cls_id, track_id in zip(boxes, clss, track_ids):
                # 1. Get the raw name (The "Guess")
                raw_name = model.names[int(cls_id)]
                
                # 2. RUN THE VOTE (The Fix)
                final_name = get_stable_name(track_id, raw_name)

                # 3. Draw the result
                x1, y1, x2, y2 = map(int, box)
                
                # Color Logic
                if vision_utils.is_hazard(final_name):
                    color = (0, 165, 255) # Orange (Obstacle)
                    label = f"OBSTACLE: {final_name}"
                else:
                    color = (0, 255, 0) # Green (Item)
                    label = f"ITEM: {final_name}"

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Label Background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imshow('Voice Vision - Stabilized Mode', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_stable_detection()