# Evaluation/config.py

# ===== GOOGLE DRIVE PATHS =====
BASE_DATASET_PATH = "/content/drive/MyDrive/voicevision_dataset"

# English audio split across condition folders
EN_AUDIO_FOLDERS = [
    f"{BASE_DATASET_PATH}/english/clean",
    f"{BASE_DATASET_PATH}/english/noisy",
    f"{BASE_DATASET_PATH}/english/fast",
    f"{BASE_DATASET_PATH}/english/slow",
    f"{BASE_DATASET_PATH}/english/unclear",
]

# Transcripts file (filename|text)
TRANSCRIPT_FILE = f"{BASE_DATASET_PATH}/transcripts/english.txt"

# Commands CSV (id, language, text, intent)
COMMANDS_CSV = "/content/drive/MyDrive/commands_sheet.csv"

# Evaluation language
LANGUAGE = "EN"

# Whisper evaluation parameters
WHISPER_LANGUAGE = "en"
BEAM_SIZE = 5

# Enable noise filtering ONLY for noisy / unclear evaluation
APPLY_NOISE_FILTER = False   # Set False when evaluating clean audio

# Change this depending on phase:
# Phase-1 → ["clean","fast","slow"]
# Phase-2 → ["noisy","unclear"]
EVAL_CONDITIONS =  ["clean","fast","slow"]