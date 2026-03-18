from gtts import gTTS
import io
import socket


class TextToSpeech:

    def __init__(self):
        # Nothing to init for now, just a placeholder
        pass


    def _is_connected(self):
        # Quick check: can we ping Google's DNS? If not, gTTS won't work.
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=2)
            return True
        except:
            return False


    def speak_to_bytes(self, text, lang_code="en-IN"):
        """
        Convert text to speech and return audio bytes.
        Mobile app will play the audio.
        """

        if not text:
            return None

        print(f"🔊 Generating speech: {text}")

        # Extract just 'en' from 'en-IN' for gTTS compatibility
        short_lang = lang_code.split("-")[0]

        try:
            # Check internet first so the app doesn't hang waiting for a timeout
            if not self._is_connected():
                print("⚠ Internet required for gTTS")
                return None

            # Hit the Google Translate TTS API
            tts = gTTS(text=text, lang=short_lang, slow=False)

            # Use a memory buffer instead of saving a physical .mp3 file
            audio_buffer = io.BytesIO()

            # Write the audio data into that buffer
            tts.write_to_fp(audio_buffer)

            # Move the pointer back to the start so we can read the whole thing
            audio_buffer.seek(0)

            # Return the raw audio bytes to be sent to the mobile app
            return audio_buffer.read()

        except Exception as e:

            print(f"TTS generation failed: {e}")

            return None