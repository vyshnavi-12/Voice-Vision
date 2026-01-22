import speech_recognition as sr
from faster_whisper import WhisperModel
import io
import os

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
        # We turn OFF dynamic adjustment to prevent the mic from becoming "deaf" in loud rooms.
        # Instead, we set a stable threshold based on actual room noise.
        self.recognizer.dynamic_energy_threshold = False 
        self.recognizer.pause_threshold = 0.8  # Wait 0.8s of silence before considering command "done"
        
        # Default Language
        self.current_lang_code = 'en-IN' 
        self.lang_names = {'en-IN': "English", 'te-IN': "Telugu", 'hi-IN': "Hindi"}

        # --- AUTO-CALIBRATION ---
        print(" 🔊 Calibrating microphone for background noise... (Please be quiet)")
        try:
            with sr.Microphone() as source:
                # Listen to the room for 1 second to set the baseline
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # FORCE SENSITIVITY LIMITS
                # If the room is super quiet, don't drop below 400 (prevents picking up breathing)
                if self.recognizer.energy_threshold < 400:
                    self.recognizer.energy_threshold = 400
                
                # If the room is super loud, don't go above 2000 (prevents being deaf to voice)
                if self.recognizer.energy_threshold > 2000:
                    self.recognizer.energy_threshold = 2000
                    
            print(f" ✅ Microphone Calibrated. Noise Threshold: {self.recognizer.energy_threshold}")
        except Exception as e:
            print(f" ⚠️ Calibration failed (using defaults): {e}")
            self.recognizer.energy_threshold = 500

        print("-------------------------------------------------------")
        print(f" 🎙️ SYSTEM READY. Speaking: {self.lang_names[self.current_lang_code]}")

    def listen(self):
        with sr.Microphone() as source:
            print(f"\n[+] Listening ({self.lang_names[self.current_lang_code]})...")
            try:
                # phrase_time_limit=10 prevents getting stuck recording forever if noise continues
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
                return audio
            except sr.WaitTimeoutError:
                return None

    def process_command(self, text):
        """
        Checks if the text is a 'Switch Language' command.
        """
        text = text.lower().strip()
        
        # ==========================================
        # 1. COMMAND: "SWITCH TO TELUGU"
        # ==========================================
        te_cmds = [
            "switch to telugu", "change to telugu", "speak in telugu", 
            "telugu mode", "enable telugu", "set language to telugu",
            "తెలుగు", "మాట్లాడు", "మార్చు", "స్విచ్ టు తెలుగు", "తెలుగు మోడ్"
        ]
        
        # ==========================================
        # 2. COMMAND: "SWITCH TO HINDI"
        # ==========================================
        hi_cmds = [
            "switch to hindi", "change to hindi", "speak in hindi", 
            "hindi mode", "enable hindi", "set language to hindi",
            "हिंदी", "हिन्दी", "स्विच टू हिंदी", "हिंदी मोड", "हिंदी में",
            "హిందీ", "హింది", "స్విచ్ టు హిందీ", "హిందీలో"
        ]
        
        # ==========================================
        # 3. COMMAND: "SWITCH TO ENGLISH"
        # ==========================================
        en_cmds = [
            "switch to english", "change to english", "speak in english", 
            "english mode", "enable english", "set language to english",
            "इंग्लिश", "अंग्रेजी", "स्विच टू इंग्लिश", "इंग्लिश मोड",
            "ఇంగ్లీష్", "ఆంగ్లం", "స్విచ్ టు ఇంగ్లీష్", "ఇంగ్లీష్ మోడ్"
        ]

        # --- EXECUTION LOGIC ---
        if any(cmd in text for cmd in te_cmds):
            if self.current_lang_code != 'te-IN':
                self.current_lang_code = 'te-IN'
                print(f"   >>> 🔄 COMMAND DETECTED: Switching to TELUGU (te-IN) <<<")
                return True 
        
        if any(cmd in text for cmd in hi_cmds):
            if self.current_lang_code != 'hi-IN':
                self.current_lang_code = 'hi-IN'
                print(f"   >>> 🔄 COMMAND DETECTED: Switching to HINDI (hi-IN) <<<")
                return True
                
        if any(cmd in text for cmd in en_cmds):
            if self.current_lang_code != 'en-IN':
                self.current_lang_code = 'en-IN'
                print(f"   >>> 🔄 COMMAND DETECTED: Switching to ENGLISH (en-IN) <<<")
                return True

        return False

    def transcribe(self, audio):
        if audio is None: return ""
        
        # --- PHASE 1: ONLINE (Google) ---
        try:
            # Get Raw Text
            raw_text = self.recognizer.recognize_google(audio, language=self.current_lang_code)
            
            # Check for command BEFORE printing
            if self.process_command(raw_text):
                return "" 
            
            print(f"✅ ONLINE ({self.lang_names[self.current_lang_code]}): {raw_text}")
            return raw_text.lower()

        except sr.UnknownValueError:
            pass 
        except sr.RequestError:
            print("⚠️ Internet down. Switching to Offline...")
        except Exception:
            pass # Keep silent on small errors

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
                    vad_filter=True, # Voice Activity Detection filters out silence/noise
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
        
        return ""

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