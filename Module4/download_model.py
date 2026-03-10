from sentence_transformers import SentenceTransformer
import os

# Folder where model will be stored
MODEL_DIR = "models/intent_model"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Downloading intent model...")

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)

model.save(MODEL_DIR)

print("Model downloaded and saved to:", MODEL_DIR)