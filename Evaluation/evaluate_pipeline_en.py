# Evaluation/evaluate_pipeline_en.py
import pandas as pd
import os

from config import (
    COMMANDS_CSV,
    LANGUAGE,
    EN_AUDIO_FOLDERS,
    EVAL_CONDITIONS,
    APPLY_NOISE_FILTER
)

from metrics import compute_intent_metrics
from stt_engine import HybridSTT
from intent_parser import IntentParser

# Initialize models
stt = HybridSTT()
parser = IntentParser()

df = pd.read_csv(COMMANDS_CSV)

# -------------------------------------------------
# Filter only selected language (EN)
# -------------------------------------------------
df = df[df["Language"] == LANGUAGE]

y_true = []
y_pred = []

print(f"\nEvaluating PIPELINE for conditions: {EVAL_CONDITIONS}")
print(f"Noise Filtering Enabled: {APPLY_NOISE_FILTER}")

for _, row in df.iterrows():
    cmd_id = row["ID"]
    true_intent = row["Intent"]

    # -------------------------------------------------
    # Evaluate across selected audio conditions
    # -------------------------------------------------
    for condition in EVAL_CONDITIONS:
        filename = f"{cmd_id}_{LANGUAGE}_{condition}.wav"
        audio_path = None

        # Search across dataset folders
        for folder in EN_AUDIO_FOLDERS:
            possible = os.path.join(folder, filename)
            if os.path.exists(possible):
                audio_path = possible
                break

        if audio_path is None:
            continue

        # -------------------------------------------------
        # STT Processing
        # -------------------------------------------------
        transcript = stt.transcribe_file_eval(audio_path)

        # -------------------------------------------------
        # *** KEY CHANGE: Use parse_noisy() for noisy/unclear ***
        # -------------------------------------------------
        if condition in ["noisy", "unclear"]:
            pred_intent = parser.parse_noisy(transcript)  # ← NOISY MODE
            print(f"🔊 NOISY MODE: {condition}")
        else:
            pred_intent = parser.parse(transcript)        # ← NORMAL MODE

        print(f"\nAUDIO : {filename}")
        print(f"TEXT  : {transcript}")
        print(f"TRUE  : {true_intent}")
        print(f"PRED  : {pred_intent}")

        y_true.append(true_intent)
        y_pred.append(pred_intent)

# -------------------------------------------------
# Compute metrics with condition breakdown
# -------------------------------------------------
print(f"\n{'='*60}")
print(f"PIPELINE RESULTS - EVAL CONDITIONS: {EVAL_CONDITIONS}")
compute_intent_metrics(y_true, y_pred)
