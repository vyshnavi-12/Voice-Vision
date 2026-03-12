from sentence_transformers import SentenceTransformer
import os

# Get project root (folder where this script is located)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Go one level up if script is inside a module folder
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

# Models folder
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "paraphrase-multilingual-mpnet-base-v2")

# Create folder if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

print("Downloading intent model...")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

# Save model locally
model.save(MODEL_DIR)

print("Model downloaded and saved to:", MODEL_DIR)