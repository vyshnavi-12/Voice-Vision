# Evaluation/stt_engine.py

from faster_whisper import WhisperModel
import torch


OFFLINE_MODEL_SIZE = "small"


class HybridSTT:
    """
    Evaluation-only STT Engine.
    Uses ONLY Whisper.
    Forces English transcription.
    """

    def __init__(self):
        print("-------------------------------------------------------")
        print(f"Loading Whisper Model ({OFFLINE_MODEL_SIZE})")

        # Select device automatically
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # safer compute type for Colab
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        print(f"Device: {self.device} | compute: {self.compute_type}")

        # Load Whisper model
        self.offline_model = WhisperModel(
            OFFLINE_MODEL_SIZE,
            device=self.device,
            compute_type=self.compute_type
        )

        print("Model Ready.")
        print("-------------------------------------------------------")

    def transcribe_file_eval(self, file_path):
        """
        Pure transcription for evaluation dataset.
        English is forced (no auto language detection).
        """

        try:
            segments, info = self.offline_model.transcribe(
                file_path,
                beam_size=5,
                language="en",              # 🔴 Force English decoding
                task="transcribe",          # Do NOT translate
                vad_filter=True,
                condition_on_previous_text=False
            )

            text = " ".join([seg.text for seg in segments])

            return text.lower().strip()

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return ""