import speech_recognition as sr
from faster_whisper import WhisperModel
import io
import os
import noise_filtering
import torch

# Configuration for offline Whisper model size
OFFLINE_MODEL_SIZE = "small"

class HybridSTT:
    def __init__(self):
        """
        Initializes the hybrid Speech-to-Text engine.
        Uses online Google STT first, then falls back
        to offline Whisper if needed.
        """
        print("-------------------------------------------------------")
        print(f" ⏳ Loading Offline Brain ('{OFFLINE_MODEL_SIZE}')...")

        # Automatically choose GPU or CPU for Whisper
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f" 🚀 Whisper device: {device} | compute_type: {compute_type}")

        # Load Whisper offline model safely
        try:
            self.offline_model = WhisperModel(
                OFFLINE_MODEL_SIZE,
                device=device,
                compute_type=compute_type
            )
            print(" ✅ Offline Model Ready.")
        except Exception as e:
            print(f" ❌ ERROR loading offline model: {e}")
            self.offline_model = None

        # Initialize Google Speech Recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold = 0.8

        # Default language and supported languages
        self.current_lang_code = 'en-IN'
        self.lang_names = {
            'en-IN': "English",
            'te-IN': "Telugu",
            'hi-IN': "Hindi"
        }

        # Fixed energy threshold for stable background noise handling
        self.recognizer.energy_threshold = 500

        print("-------------------------------------------------------")
        print(f" 🎙️ SYSTEM READY. Current Language: {self.lang_names[self.current_lang_code]}")


    def listen(self):
        """
        Listens to microphone input and returns raw audio.
        Returns None if no speech is detected within timeout.
        """
        with sr.Microphone() as source:
            print(f"\n[+] Listening ({self.lang_names[self.current_lang_code]})...")
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
        Converts audio input to text.
        First attempts online recognition, then falls back
        to offline Whisper if online fails.
        """
        if audio is None:
            return ""

        # Apply noise filtering before transcription
        print(" ✨ Filtering audio noise...")
        try:
            clean_bytes = noise_filtering.clean_audio_data(audio)
            audio = sr.AudioData(clean_bytes, 16000, 2)
        except Exception as e:
            print(f" ⚠️ Filter Error: {e}. Using raw audio.")

        # Attempt online STT using Google
        try:
            raw_text = self.recognizer.recognize_google(
                audio,
                language=self.current_lang_code
            )

            # Handle language-switch commands immediately
            if self.process_command(raw_text):
                return ""

            print(f"✅ ONLINE ({self.lang_names[self.current_lang_code]}): {raw_text}")
            return raw_text.lower()

        except (sr.UnknownValueError, sr.RequestError):
            pass

        # Fallback to offline Whisper STT if online fails
        if self.offline_model:
            print("⚡ Processing Offline...")
            try:
                wav_data = io.BytesIO(audio.get_wav_data())
                whisper_lang_hint = self.current_lang_code.split('-')[0]

                segments, info = self.offline_model.transcribe(
                    wav_data,
                    beam_size=1,
                    language=whisper_lang_hint,
                    vad_filter=True
                )

                raw_text = " ".join([segment.text for segment in segments])

                # Handle language-switch commands from Whisper output
                if self.process_command(raw_text):
                    return ""

                if raw_text.strip():
                    print(f"✅ OFFLINE (Whisper): {raw_text.strip()}")
                    return raw_text.strip().lower()

            except Exception as e:
                print(f"❌ Offline Error: {e}")

        return ""


    def process_command(self, text):
        """
        Detects language-switch commands using substring matching.
        Commands are written in multiple scripts to allow
        cross-language switching.
        """
        t = text.lower().strip()

        # Commands that switch system language to Telugu
        te_cmds = [
            "switch to telugu", "change to telugu", "telugu mode", "speak in telugu",
            "तेलुगु में बोलो", "तेलुगु मोड लगाओ", "तेलुगु में बात करो", "तेलुगु भाषा"
        ]

        # Commands that switch system language to Hindi
        hi_cmds = [
            "switch to hindi", "change to hindi", "hindi mode", "speak in hindi",
            "హిందీలో మాట్లాడు", "హిందీకి మార్చు", "హిందీ మోడ్", "హిందీ భాష"
        ]

        # Commands that switch system language to English
        en_cmds = [
            "ఇంగ్లీష్ లో మాట్లాడు", "ఇంగ్లీష్ కి మార్చు", "ఇంగ్లీష్ మోడ్", "ఆంగ్లంలో మాట్లాడు",
            "अंग्रेजी में बोलो", "इंग्लिश मोड", "इंग्लिश में बात करो", "इंग्लिश लगाओ"
        ]

        # Perform language switching based on detected command
        if any(cmd in t for cmd in te_cmds):
            if self.current_lang_code != 'te-IN':
                self.current_lang_code = 'te-IN'
                print(" >>> 🔄 Switch Detected: TELUGU")
                return True

        if any(cmd in t for cmd in hi_cmds):
            if self.current_lang_code != 'hi-IN':
                self.current_lang_code = 'hi-IN'
                print(" >>> 🔄 Switch Detected: HINDI")
                return True

        if any(cmd in t for cmd in en_cmds):
            if self.current_lang_code != 'en-IN':
                self.current_lang_code = 'en-IN'
                print(" >>> 🔄 Switch Detected: ENGLISH")
                return True

        return False


if __name__ == "__main__":
    """
    Simple test loop for microphone input and transcription.
    Stops when user says 'exit' or 'stop'.
    """
    engine = HybridSTT()

    while True:
        try:
            audio_input = engine.listen()
            if audio_input:
                command = engine.transcribe(audio_input)
                if command and ("exit" in command or "stop" in command):
                    print("Stopping...")
                    break
        except KeyboardInterrupt:
            print("\nExiting...")
            break
