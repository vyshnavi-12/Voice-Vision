import cv2
import winsound
import time
from ultralytics import YOLO

model = YOLO("models/best.pt")

# Print ALL class names your model knows — check this in terminal
print("Your model's classes:")
for i, name in model.names.items():
    print(f"  {i}: {name}")

OBSTACLES = {
    "person", "dog", "cattle", "goat",
    "ambulance", "bike", "bullock-cart", "bus", "car", "crane",
    "cycle", "rikshaw", "tempo", "tractor", "truck",
    "barricade", "electricity-pole", "lamp-post", "manhole",
    "road-divider", "traffic-sign-board", "traffic-signal", "zebra-crossing",
    "chair", "door", "fence", "garbage bin", "garbage_bin", "obstacle",
    "plant", "pothole", "stairs", "table", "vehicle"
}

last_beep = 0
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.25, verbose=False)
    too_close = False

    for box in results[0].boxes:
        cls_name = model.names[int(box.cls[0])].lower()
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        box_h = y2 - y1
        distance = (1.7 * 600) / box_h if box_h > 0 else 99

        is_obstacle = cls_name in OBSTACLES

        if is_obstacle:
            color = (0, 0, 255) if distance < 1.5 else (0, 165, 255) if distance < 3 else (0, 255, 0)
            label = f"{cls_name} {distance:.1f}m"
            if distance < 3:
                too_close = True
        else:
            color = (160, 160, 160)
            label = cls_name

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if too_close and time.time() - last_beep > 2:
        winsound.Beep(800, 300)
        last_beep = time.time()

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()