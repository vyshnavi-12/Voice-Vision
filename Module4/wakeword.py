import pvporcupine
import struct
import os
import numpy as np

class WakeWordListener:
    def __init__(self):
        self.ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

        self.KEYWORD_PATH = os.path.join(
            PROJECT_ROOT,
            "models",
            "Hello-Vision_en_windows_v4_0_0.ppn"
        )

        if not os.path.exists(self.KEYWORD_PATH):
            raise FileNotFoundError(f"❌ Error: Could not find '{self.KEYWORD_PATH}'")

        try:
            self.porcupine = pvporcupine.create(
                access_key=self.ACCESS_KEY,
                keyword_paths=[self.KEYWORD_PATH]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Porcupine: {e}")

    def process_audio(self, pcm_bytes):
        # 1. Safety check: Ensure even number of bytes for int16 conversion
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]

        # 2. Convert raw bytes to 16-bit PCM (Shorts)
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        
        # 3. Slide through the audio in 512-sample frames
        frame_length = self.porcupine.frame_length
        
        for i in range(0, len(pcm) - frame_length, frame_length):
            frame = pcm[i : i + frame_length]
            result = self.porcupine.process(frame)
            if result >= 0:
                print("🔥 [WAKEWORD] Match found in audio stream!")
                return True
        return False

    def cleanup(self):
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()