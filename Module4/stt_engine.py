# stt_engine.py

import speech_recognition as sr
from faster_whisper import WhisperModel
import io
import torch


OFFLINE_MODEL_SIZE = "small"


class WhisperSTT:
    """
    Real-time Whisper STT Engine.
    - Uses ONLY Whisper (no Google STT)
    - No external noise filtering (Whisper handles robustness)
    - Language is forced externally
    - Default language = English ("en")
    """

    def __init__(self, language="en"):
        print("-------------------------------------------------------")
        print(f"⏳ Loading Whisper Model ({OFFLINE_MODEL_SIZE})")

        # Store current language
        self.language = language

        # Select device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        print(f"🚀 Device: {self.device} | compute: {self.compute_type}")

        # Load Whisper
        self.model = WhisperModel(
            OFFLINE_MODEL_SIZE,
            device=self.device,
            compute_type=self.compute_type
        )

        # Speech recognizer (only for capturing mic audio)
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 500
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold = 0.8

        print(f"🎙️ STT Ready | Forced Language: {self.language}")
        print("-------------------------------------------------------")


    def set_language(self, new_language):
        """
        Allows external module (intent parser) to change language.
        Example: "en", "hi", "te"
        """
        self.language = new_language
        print(f"🌐 Language switched to: {self.language}")


    def listen(self):
        """
        Captures microphone audio.
        """
        with sr.Microphone() as source:
            print(f"\n[+] Listening ({self.language})...")
            try:
                audio = self.recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=10
                )
                return audio
            except sr.WaitTimeoutError:
                return None


    def transcribe(self, audio):
        """
        Transcribes real-time microphone audio using Whisper.
        """

        if audio is None:
            return ""

        try:
            print("⚡ Transcribing with Whisper...")

            wav_data = io.BytesIO(audio.get_wav_data())

            segments, info = self.model.transcribe(
                wav_data,
                beam_size=5,
                language=self.language,   # 🔴 Forced Language
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


if __name__ == "__main__":

    engine = WhisperSTT(language="en")  # Default English

    while True:
        try:
            audio_input = engine.listen()
            if audio_input:
                result = engine.transcribe(audio_input)

                # Example manual exit
                if result and ("exit" in result or "stop" in result):
                    print("Stopping...")
                    break

        except KeyboardInterrupt:
            print("\nExiting...")
            break