# Module4/tts_engine.py
from gtts import gTTS
import pygame
import os
import time
import subprocess # <--- NEW: Allows running Windows commands directly

class TextToSpeech:
    def __init__(self):
        # We don't need to init anything for the new offline method
        pass

    def _speak_offline_windows(self, text):
        """
        Uses Windows PowerShell to speak. 
        Reliable because it runs outside Python's audio system.
        """
        print(f"   [System TTS] Speaking: '{text}'")
        
        # Escape quotes in text to prevent errors (e.g., "don't" -> "don''t")
        safe_text = text.replace("'", "''").replace('"', '')
        
        # The PowerShell Command to speak
        command = f'PowerShell -Command "Add-Type –AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{safe_text}\');"'
        
        try:
            # Run the command and wait for it to finish
            # creationflags=0x08000000 hides the console window popup
            subprocess.run(command, shell=True, creationflags=0x08000000)
        except Exception as e:
            print(f" ❌ System TTS Failed: {e}")

    def speak(self, text, lang_code="en-IN"):
        if not text: return
        print(f" 🔊 Speaking: {text}")

        # --- ATTEMPT 1: ONLINE (Google TTS) ---
        try:
            # Initialize Pygame Mixer ONLY when needed
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            short_lang = lang_code.split('-')[0]
            filename = "temp_voice.mp3"
            
            # Generate
            tts = gTTS(text=text, lang=short_lang, slow=False)
            tts.save(filename)

            # Play
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
            # Cleanup
            pygame.mixer.music.unload()
            try:
                os.remove(filename)
            except: pass
            return  # Success!

        except Exception as e:
            print(f" ⚠️ Online TTS Failed: {e}")
            print(" ⚡ Switching to System TTS...")

        # --- ATTEMPT 2: OFFLINE (Windows System) ---
        try:
            # 1. Release Pygame lock just in case
            if pygame.mixer.get_init():
                pygame.mixer.quit()
            
            # 2. Use the new robust Windows method
            self._speak_offline_windows(text)
            
        except Exception as e:
            print(f" ❌ Offline Critical Failure: {e}")