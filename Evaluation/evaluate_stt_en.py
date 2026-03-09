# Evaluation/evaluate_stt_en.py

from config import TRANSCRIPT_FILE, EVAL_CONDITIONS, APPLY_NOISE_FILTER
from audio_utils import find_audio_file
from metrics import compute_wer
from stt_engine import HybridSTT

# NEW: noise filtering (used only when enabled)
# from noise_filtering import clean_audio_for_stt

engine = HybridSTT()

refs = []
hyps = []

print(f"\nEvaluating STT for conditions: {EVAL_CONDITIONS}")
print(f"Noise Filtering Enabled: {APPLY_NOISE_FILTER}")

with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    fname, gt_text = line.strip().split("|")

    # -------------------------------------------------
    # Filter dataset based on selected conditions
    # -------------------------------------------------
    if not any(cond in fname.lower() for cond in EVAL_CONDITIONS):
        continue

    audio_path = find_audio_file(fname)

    # -------------------------------------------------
    # Apply noise filtering ONLY for noisy/unclear phase
    # -------------------------------------------------
    # if APPLY_NOISE_FILTER:
    #     audio_path = clean_audio_for_stt(audio_path)

    pred = engine.transcribe_file_eval(audio_path)

    print(f"\nAUDIO: {fname}")
    print(f"GT   : {gt_text}")
    print(f"PRED : {pred}")

    refs.append(gt_text.lower())
    hyps.append(pred.lower())

compute_wer(refs, hyps)