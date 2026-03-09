# Evaluation/audio_utils.py
import os
from config import EN_AUDIO_FOLDERS

def find_audio_file(filename):
    """
    Search audio file across clean/noisy/fast/slow/unclear folders.
    """
    for folder in EN_AUDIO_FOLDERS:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"{filename} not found in dataset folders.")