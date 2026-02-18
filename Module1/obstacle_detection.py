import cv2
import os
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

# -------------------------------
# Beep Function (Windows + fallback)
# -------------------------------
def beep():
    try:
        import winsound
        winsound.Beep(1000, 200)
    except:
        print("\a", end="", flush=True)


# -------------------------------
# Distance Parameters (Adjust later if needed)
# -------------------------------
KNOWN_HEIGHT = 1.0   # meters
FOCAL_LENGTH = 500   # calibration constant
SAFE_DISTANCE = 1.5  # meters


# -------------------------------
# Load YOLO Model
# -------------------------------
model = YOLO("models/best.pt")


# -------------------------------
# Proper File Dialog (Important Fix)
# -------------------------------
def choose_file(file_type):
    root = tk.Tk()
    root.withdraw()                 # Hide root window
    root.update()
    root.attributes('-topmost', True)  # Bring dialog to front

    if file_type == "image":
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")],
            parent=root
        )
    else:
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")],
            parent=root
        )

    root.destroy()   # VERY important on Windows
    return file_path


# -------------------------------
# Core Detection Logic
# -------------------------------
def process_frame(frame):
    results = model(frame, verbose=False)[0]

    if results.boxes is None:
        return frame

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        pixel_height = y2 - y1
        distance = (KNOWN_HEIGHT * FOCAL_LENGTH) / pixel_height if pixel_height > 0 else 0

        critical = distance <= SAFE_DISTANCE

        if critical:
            print(f"[CRITICAL] {label} at {distance:.2f} m")
            beep()
        else:
            print(f"[SAFE] {label} at {distance:.2f} m")

        color = (0, 0, 255) if critical else (0, 255, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {distance:.2f}m",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


# -------------------------------
# Image Mode
# -------------------------------
def run_image():
    path = choose_file("image")
    if not path:
        print("No image selected.")
        return

    image = cv2.imread(path)
    output = process_frame(image)

    save_path = "output_detected.jpg"
    cv2.imwrite(save_path, output)

    print(f"Saved → {save_path}")

    cv2.imshow("Obstacle Detection - Image", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------------
# Video Mode
# -------------------------------
def run_video():
    path = choose_file("video")
    if not path:
        print("No video selected.")
        return

    cap = cv2.VideoCapture(path)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter("output_detected.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))

    print("Processing video... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        output = process_frame(frame)
        out.write(output)

        cv2.imshow("Obstacle Detection - Video", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Saved → output_detected.mp4")


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
        cv2.imshow("Obstacle Detection - Webcam", output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# -------------------------------
# Main Menu
# -------------------------------
if __name__ == "__main__":
    while True:
        print("\nChoose Input Source:")
        print("1 → Image")
        print("2 → Video")
        print("3 → Webcam")
        print("4 → Exit")

        choice = input("Enter choice (1/2/3/4): ").strip()

        if choice == "1":
            run_image()

        elif choice == "2":
            run_video()

        elif choice == "3":
            run_webcam()

        elif choice == "4":
            print("Exiting...")
            break

        else:
            print("Invalid choice. Try again.")