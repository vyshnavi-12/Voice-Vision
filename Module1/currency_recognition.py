import cv2
from ultralytics import YOLO
import time

# -----------------------------
# LOAD TRAINED MODEL
# -----------------------------
model = YOLO("models/currency_best.pt")   # your trained model

# -----------------------------
# OPEN CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Currency detection started... Press 'q' to quit.")

# To avoid repeating speech/log every frame
last_spoken = ""
last_time = 0
delay = 2  # seconds between announcements

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -----------------------------
    # RUN DETECTION
    # -----------------------------
    results = model(frame, conf=0.6, verbose=False)

    detected_currency = None

    for r in results:
        boxes = r.boxes

        if boxes is None:
            continue

        for box in boxes:
            # Get class name
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            # Convert class name → readable ₹ label
            currency_label = f"₹{label}"

            detected_currency = currency_label

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label text
            cv2.putText(frame, currency_label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2)

    # -----------------------------
    # PRINT RESULT (CONTROLLED)
    # -----------------------------
    current_time = time.time()

    if detected_currency and (detected_currency != last_spoken or current_time - last_time > delay):
        print(f"Detected Currency: {detected_currency}")
        last_spoken = detected_currency
        last_time = current_time

    # -----------------------------
    # SHOW FRAME
    # -----------------------------
    cv2.imshow("Currency Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()