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
        self.recognizer.energy_threshold = 300  
        self.recognizer.dynamic_energy_threshold = False
        
        # Default Language
        self.current_lang_code = 'en-IN' 
        self.lang_names = {'en-IN': "English", 'te-IN': "Telugu", 'hi-IN': "Hindi"}
        
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

    def process_command(self, text):
        """
        Checks if the text is a 'Switch Language' command.
        Includes extensive lists for English, Hindi, and Telugu in all scripts.
        """
        text = text.lower().strip()
        
        # ==========================================
        # 1. COMMAND: "SWITCH TO TELUGU"
        # ==========================================
        te_cmds = [
            # --- English Phrasing ---
            "switch to telugu", "change to telugu", "speak in telugu", 
            "telugu mode", "enable telugu", "set language to telugu",
            "can you speak telugu", "language telugu",
            
            # --- Hindi Script (Devanagari) ---
            "तेलुगु", "तेलुगू", "स्विच टू तेलुगु", "तेलुगु में", 
            "तेलुगु मोड", "तेलुगु में बदलो", "तेलुगु लगाओ",
            
            # --- Telugu Script (Native) ---
            "తెలుగు", "మాట్లాడు", "మార్చు", "భాష మార్చు",
            "స్విచ్ టు తెలుగు", "తెలుగు మోడ్", "తెలుగులో మాట్లాడు",
            "తెలుగులోకి మార్చు", "తెలుగు భాష", "తెలుగు ఎనేబుల్ చెయ్యి"
        ]
        
        # ==========================================
        # 2. COMMAND: "SWITCH TO HINDI"
        # ==========================================
        hi_cmds = [
            # --- English Phrasing ---
            "switch to hindi", "change to hindi", "speak in hindi", 
            "hindi mode", "enable hindi", "set language to hindi",
            "can you speak hindi", "language hindi",
            
            # --- Hindi Script (Devanagari) ---
            "हिंदी", "हिन्दी", "स्विच टू हिंदी", "हिंदी मोड", "हिंदी में",
            "हिंदी में बदलो", "हिंदी लगाओ", "हिंदी में बात करो",
            
            # --- Telugu Script (Native) ---
            "హిందీ", "హింది", "స్విచ్ టు హిందీ", "హిందీలో", 
            "హిందీ మోడ్", "హిందీలోకి మార్చు", "హిందీ భాష", "హిందీ మాట్లాడు"
        ]
        
        # ==========================================
        # 3. COMMAND: "SWITCH TO ENGLISH"
        # ==========================================
        en_cmds = [
            # --- English Phrasing ---
            "switch to english", "change to english", "speak in english", 
            "english mode", "enable english", "set language to english",
            "normal mode", "default mode", "language english",
            
            # --- Hindi Script (Devanagari) ---
            "इंग्लिश", "अंग्रेजी", "स्विच टू इंग्लिश", "इंग्लिश मोड", "अंग्रेजी में",
            "इंग्लिश में", "अंग्रेजी मोड", "इंग्लिश लगाओ",
            
            # --- Telugu Script (Native) ---
            "ఇంగ్లీష్", "ఆంగ్లం", "స్విచ్ టు ఇంగ్లీష్", "ఇంగ్లీష్ మోడ్", "ఇంగ్లీష్ లో",
            "ఇంగ్లీష్ భాష", "ఆంగ్ల భాష", "ఇంగ్లీష్ మాట్లాడు"
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
        text = ""

        # --- PHASE 1: ONLINE (Google) ---
        try:
            # 1. Get Raw Text
            raw_text = self.recognizer.recognize_google(audio, language=self.current_lang_code)
            
            # 2. INTERCEPT: Check for command BEFORE printing
            if self.process_command(raw_text):
                return "" 
            
            print(f"✅ ONLINE ({self.lang_names[self.current_lang_code]}): {raw_text}")
            return raw_text.lower()

        except sr.UnknownValueError:
            pass 
        except sr.RequestError:
            print("⚠️ Internet down. Switching to Offline...")

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