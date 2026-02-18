import cv2
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
from object_classes import OBJECT_CLASSES

# -------------------------------
# Load General YOLO Model
# -------------------------------
model = YOLO("models/yolov8x.pt")  # pretrained COCO model
YOLO_NAMES = model.names


# -------------------------------
# File Picker (Image)
# -------------------------------
def choose_image():
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )

    root.destroy()
    return path


# -------------------------------
# Process Frame for Allowed Objects
# -------------------------------
def process_frame(frame):
    results = model(frame, verbose=False)[0]

    if results.boxes is None:
        return frame

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = YOLO_NAMES[cls_id]

        # Only keep useful interaction objects
        if label not in OBJECT_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        print(f"Detected: {label} ({conf:.2f})")

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{label}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 0, 0), 2)

    return frame


# -------------------------------
# Image Mode
# -------------------------------
def run_image():
    path = choose_image()
    if not path:
        print("No image selected.")
        return

    image = cv2.imread(path)
    output = process_frame(image)

    cv2.imwrite("object_output.jpg", output)
    print("Saved → object_output.jpg")

    cv2.imshow("Object Detection - Image", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------------
# Webcam Mode
# -------------------------------
def run_webcam():
    cap = cv2.VideoCapture(0)
    print("Webcam running... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = process_frame(frame)
        cv2.imshow("Object Detection - Webcam", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# Loop Menu
# -------------------------------
if __name__ == "__main__":
    while True:
        print("\nObject Detection Menu:")
        print("1 → Image")
        print("2 → Webcam")
        print("3 → Exit")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            run_image()
        elif choice == "2":
            run_webcam()
        elif choice == "3":
            break
        else:
            print("Invalid choice.")