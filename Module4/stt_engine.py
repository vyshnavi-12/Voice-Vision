import speech_recognition as sr
from faster_whisper import WhisperModel
import io
import os
import noise_filtering  

# --- CONFIGURATION ---
OFFLINE_MODEL_SIZE = "small"   
COMPUTE_TYPE = "int8"

class HybridSTT:
    def __init__(self):
        print("-------------------------------------------------------")
        print(f" ⏳ Loading Offline Brain ('{OFFLINE_MODEL_SIZE}')...")
        try:
            self.offline_model = WhisperModel(OFFLINE_MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)
            print(" ✅ Offline Model Ready.")
        except Exception as e:
            print(f" ❌ ERROR loading offline model: {e}")
            self.offline_model = None

        self.recognizer = sr.Recognizer()
        
        # --- NOISE CONTROL SETTINGS ---
        self.recognizer.dynamic_energy_threshold = False 
        self.recognizer.pause_threshold = 0.8  
        
        # Default Language
        self.current_lang_code = 'en-IN' 
        self.lang_names = {'en-IN': "English", 'te-IN': "Telugu", 'hi-IN': "Hindi"}

        # --- AUTO-CALIBRATION ---
        print(" 🔊 Calibrating microphone for background noise... (Please be quiet)")
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Calibration limits
                if self.recognizer.energy_threshold < 400:
                    self.recognizer.energy_threshold = 400
                if self.recognizer.energy_threshold > 2000:
                    self.recognizer.energy_threshold = 2000
                    
            print(f" ✅ Microphone Calibrated. Threshold: {self.recognizer.energy_threshold}")
        except Exception as e:
            print(f" ⚠️ Calibration failed: {e}")
            self.recognizer.energy_threshold = 500

        print("-------------------------------------------------------")
        print(f" 🎙️ SYSTEM READY. Speaking: {self.lang_names[self.current_lang_code]}")

    def listen(self):
        with sr.Microphone() as source:
            print(f"\n[+] Listening ({self.lang_names[self.current_lang_code]})...")
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                return audio
            except sr.WaitTimeoutError:
                return None

    def transcribe(self, audio):
        if audio is None: return ""

        # --- NOISE FILTERING STEP ---
        print(" ✨ Filtering audio noise...")
        try:
            clean_bytes = noise_filtering.clean_audio_data(audio)
            audio = sr.AudioData(clean_bytes, 16000, 2)
        except Exception as e:
            print(f" ⚠️ Filter Error: {e}. Using raw audio.")
        
        # --- PHASE 1: ONLINE (Google) ---
        try:
            raw_text = self.recognizer.recognize_google(audio, language=self.current_lang_code)
            if self.process_command(raw_text):
                return "" 
            
            print(f"✅ ONLINE ({self.lang_names[self.current_lang_code]}): {raw_text}")
            return raw_text.lower()

        except (sr.UnknownValueError, sr.RequestError):
            pass

        # --- PHASE 2: OFFLINE (Whisper) ---
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
                
                if self.process_command(raw_text):
                    return ""

                if raw_text.strip():
                    print(f"✅ OFFLINE (Whisper): {raw_text.strip()}")
                    return raw_text.strip().lower()

            except Exception as e:
                print(f"❌ Offline Error: {e}")
        
        return ""

    def process_command(self, text):
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

# Restoring your original test block
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