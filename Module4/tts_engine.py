from gtts import gTTS              # Online text-to-speech (Google)
import pygame                      # Audio playback
import os                          # File handling
import time                        # Timing control for playback
import subprocess                  # Run Windows system commands directly

class TextToSpeech:
    def __init__(self):
        # No initialization required for system-level TTS
        pass

    def _speak_offline_windows(self, text):
        """
        Uses Windows PowerShell Text-to-Speech.
        Very reliable since it bypasses Python audio libraries.
        """
        print(f"   [System TTS] Speaking: '{text}'")

        # Escape problematic characters for PowerShell execution
        safe_text = text.replace("'", "''").replace('"', '')

        # PowerShell command using Windows Speech Synthesizer
        command = (
            'PowerShell -Command '
            '"Add-Type –AssemblyName System.Speech; '
            '(New-Object System.Speech.Synthesis.SpeechSynthesizer)'
            f'.Speak(\'{safe_text}\');"'
        )

        try:
            # Run command silently (no popup console)
            subprocess.run(
                command,
                shell=True,
                creationflags=0x08000000
            )
        except Exception as e:
            print(f" ❌ System TTS Failed: {e}")

    def speak(self, text, lang_code="en-IN"):
        """
        Speaks the given text using online TTS first,
        then falls back to offline Windows TTS if needed.
        """
        if not text:
            return

        print(f" 🔊 Speaking: {text}")

        # --- ATTEMPT 1: ONLINE TTS (Google gTTS) ---
        try:
            # Initialize audio system only when required
            if not pygame.mixer.get_init():
                pygame.mixer.init()

            short_lang = lang_code.split('-')[0]
            filename = "temp_voice.mp3"

            # Generate speech audio file
            tts = gTTS(text=text, lang=short_lang, slow=False)
            tts.save(filename)

            # Play generated audio
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()

            # Wait until playback finishes
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)

            # Cleanup audio resources
            pygame.mixer.music.unload()
            try:
                os.remove(filename)
            except:
                pass

            return  # Online TTS successful

        except Exception as e:
            print(f" ⚠️ Online TTS Failed: {e}")
            print(" ⚡ Switching to System TTS...")

        # --- ATTEMPT 2: OFFLINE TTS (Windows System Voice) ---
        try:
            # Release pygame audio lock before system TTS
            if pygame.mixer.get_init():
                pygame.mixer.quit()

            # Speak using Windows native speech engine
            self._speak_offline_windows(text)

        except Exception as e:
            print(f" ❌ Offline Critical Failure: {e}")
