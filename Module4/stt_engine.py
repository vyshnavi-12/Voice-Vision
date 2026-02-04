import speech_recognition as sr        # Online speech recognition (Google STT)
from faster_whisper import WhisperModel # Offline Whisper-based STT
import io                               # Handle audio byte streams
import os                               # System utilities
import noise_filtering                # Custom audio noise cleanup module
import torch                           # GPU detection

# --- CONFIGURATION ---
OFFLINE_MODEL_SIZE = "small"          # Whisper model size (speed vs accuracy)

class HybridSTT:
    def __init__(self):
        print("-------------------------------------------------------")
        print(f" ⏳ Loading Offline Brain ('{OFFLINE_MODEL_SIZE}')...")

        # Detect device automatically
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        print(f" 🚀 Whisper device: {device} | compute_type: {compute_type}")

        # Initialize offline Whisper model
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

        # Initialize SpeechRecognition engine
        self.recognizer = sr.Recognizer()

        # --- NOISE CONTROL SETTINGS ---
        self.recognizer.dynamic_energy_threshold = False
        self.recognizer.pause_threshold = 0.8

        # Default language and supported languages
        self.current_lang_code = 'en-IN'
        self.lang_names = {
            'en-IN': "English",
            'te-IN': "Telugu",
            'hi-IN': "Hindi"
        }

        # --- MICROPHONE AUTO-CALIBRATION ---
        print(" 🔊 Skipping microphone calibration (file-based STT mode)")
        self.recognizer.energy_threshold = 500

        print("-------------------------------------------------------")
        print(f" 🎙️ SYSTEM READY. Speaking: {self.lang_names[self.current_lang_code]}")

    def listen(self):
        """
        Listens to microphone input and captures spoken audio.
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
        Converts audio to text using online STT first,
        then falls back to offline Whisper if needed.
        """
        if audio is None:
            return ""

        # --- FILE HANDLING ADDITION ---
        # If 'audio' is a string path, convert it to AudioData
        if isinstance(audio, str):
            if not os.path.exists(audio):
                print(f" ❌ Audio file not found: {audio}")
                return ""
            with sr.AudioFile(audio) as source:
                audio = self.recognizer.record(source)

        # --- NOISE FILTERING STEP ---
        print(" ✨ Filtering audio noise...")
        try:
            clean_bytes = noise_filtering.clean_audio_data(audio)
            audio = sr.AudioData(clean_bytes, 16000, 2)
        except Exception as e:
            print(f" ⚠️ Filter Error: {e}. Using raw audio.")

        # --- PHASE 1: ONLINE STT (Google) ---
        try:
            raw_text = self.recognizer.recognize_google(
                audio,
                language=self.current_lang_code
            )

            # Handle language-switch commands
            if self.process_command(raw_text):
                return ""

            print(f"✅ ONLINE ({self.lang_names[self.current_lang_code]}): {raw_text}")
            return raw_text.lower()

        except (sr.UnknownValueError, sr.RequestError):
            pass

        # --- PHASE 2: OFFLINE STT (Whisper) ---
        if self.offline_model:
            print("⚡ Processing Offline...")
            try:
                wav_data = io.BytesIO(audio.get_wav_data())
                whisper_lang_hint = self.current_lang_code.split('-')[0]

                segments, info = self.offline_model.transcribe(
                    wav_data,
                    beam_size=1,
                    language=whisper_lang_hint,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    condition_on_previous_text=False
                )

                raw_text = " ".join([segment.text for segment in segments])

                # Handle language-switch commands
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
        Detects spoken language-switch commands.
        """
        text = text.lower().strip()

        te_cmds = ["switch to telugu", "change to telugu", "తెలుగు"]
        hi_cmds = ["switch to hindi", "change to hindi", "हिंदी"]
        en_cmds = ["switch to english", "change to english", "english mode"]

        if any(cmd in text for cmd in te_cmds):
            if self.current_lang_code != 'te-IN':
                self.current_lang_code = 'te-IN'
                print(f" >>> 🔄 Switching to TELUGU")
                return True

        if any(cmd in text for cmd in hi_cmds):
            if self.current_lang_code != 'hi-IN':
                self.current_lang_code = 'hi-IN'
                print(f" >>> 🔄 Switching to HINDI")
                return True

        if any(cmd in text for cmd in en_cmds):
            if self.current_lang_code != 'en-IN':
                self.current_lang_code = 'en-IN'
                print(f" >>> 🔄 Switching to ENGLISH")
                return True

        return False

# --- TESTING BLOCK ---
if __name__ == "__main__":
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
