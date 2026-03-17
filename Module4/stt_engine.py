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
        print(f"⏳ Loading Whisper Model ({OFFLINE_MODEL_SIZE})")

        self.language = language

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        print(f"🚀 Device: {self.device} | compute: {self.compute_type}")

        self.model = WhisperModel(
            OFFLINE_MODEL_SIZE,
            device=self.device,
            compute_type=self.compute_type
        )

        print(f"🎙️ STT Ready | Forced Language: {self.language}")
        print("-------------------------------------------------------")


    def set_language(self, new_language):

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

            wav_data = io.BytesIO(audio_bytes)

            segments, info = self.model.transcribe(
                wav_data,
                beam_size=5,
                language=self.language,
                task="transcribe",
                vad_filter=True,
                condition_on_previous_text=False
            )

            text = " ".join([segment.text for segment in segments])

            if text.strip():
                print(f"✅ Whisper Output: {text.strip()}")
                return text.strip().lower()

        except Exception as e:
            print(f"❌ Whisper Error: {e}")

        return ""