# stt_engine.py

from faster_whisper import WhisperModel
import io
import torch


OFFLINE_MODEL_SIZE = "small"


class WhisperSTT:
    """
    Whisper STT Engine for backend.
    Receives audio bytes from mobile app instead of microphone.
    """

    def __init__(self, language="en"):

        print("-------------------------------------------------------")
        # Just a log to show which model size we are loading (small is the sweet spot)
        print(f"⏳ Loading Whisper Model ({OFFLINE_MODEL_SIZE})")

        self.language = language

        # Logic to check if we have a GPU (CUDA) or just the CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # If GPU, use float16 for speed; if CPU, use int8 to save memory
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        print(f"🚀 Device: {self.device} | compute: {self.compute_type}")

        # Initializing the actual Faster-Whisper model
        self.model = WhisperModel(
            OFFLINE_MODEL_SIZE,
            device=self.device,
            compute_type=self.compute_type
        )

        print(f"🎙️ STT Ready | Forced Language: {self.language}")
        print("-------------------------------------------------------")


    def set_language(self, new_language):
        # Dynamically change language (good for multi-lingual support)
        self.language = new_language
        print(f"🌐 Language switched to: {self.language}")


    def transcribe(self, audio_bytes):
        """
        Transcribe audio received from mobile app.
        audio_bytes = WAV audio bytes
        """

        if not audio_bytes:
            return ""

        try:

            print("⚡ Transcribing with Whisper...")

            # Wrap the raw bytes into a file-like object so Whisper can read it
            wav_data = io.BytesIO(audio_bytes)

            # The actual conversion happens here
            segments, info = self.model.transcribe(
                wav_data,
                beam_size=5, # Higher beam size = better accuracy but slower
                language=self.language,
                task="transcribe",
                vad_filter=True, # This ignores silences automatically
                condition_on_previous_text=False, # Keeps each command independent
                initial_prompt="chair, door, table, bottle, cup, phone, book, bag, person, exit"
            )

            # Join all detected text fragments into one single string
            text = " ".join([segment.text for segment in segments])

            if text.strip():
                print(f"✅ Whisper Output: {text.strip()}")
                return text.strip().lower()

        except Exception as e:
            print(f"❌ Whisper Error: {e}")

        return ""