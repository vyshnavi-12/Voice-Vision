from gtts import gTTS
import io
import socket


class TextToSpeech:

    def __init__(self):
        pass


    def _is_connected(self):

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

        short_lang = lang_code.split("-")[0]

        try:

            if not self._is_connected():
                print("⚠ Internet required for gTTS")
                return None

            tts = gTTS(text=text, lang=short_lang, slow=False)

            audio_buffer = io.BytesIO()

            tts.write_to_fp(audio_buffer)

            audio_buffer.seek(0)

            return audio_buffer.read()

        except Exception as e:

            print(f"TTS generation failed: {e}")

            return None