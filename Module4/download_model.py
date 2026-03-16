from sentence_transformers import SentenceTransformer
from ultralytics import YOLO
import os

# Get project root (folder where this script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one level up if script is inside a module folder
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Models folder
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

# =========================
# Download Intent Model
# =========================

INTENT_MODEL_DIR = os.path.join(MODELS_ROOT, "paraphrase-multilingual-mpnet-base-v2")

os.makedirs(INTENT_MODEL_DIR, exist_ok=True)

print("Downloading intent model...")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Save model locally
model.save(INTENT_MODEL_DIR)

print("Intent model saved to:", INTENT_MODEL_DIR)


# =========================
# Download YOLOv8x Model
# =========================

YOLO_DIR = os.path.join(MODELS_ROOT, "yolo")

os.makedirs(YOLO_DIR, exist_ok=True)

YOLO_MODEL_PATH = os.path.join(YOLO_DIR, "yolov8x.pt")

print("Downloading YOLOv8x model...")

# This automatically downloads the model
yolo_model = YOLO("yolov8x.pt")

# Save it to models folder
yolo_model.save(YOLO_MODEL_PATH)

print("YOLOv8x model saved to:", YOLO_MODEL_PATH)


print("\nAll models downloaded successfully.")