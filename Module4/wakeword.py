import pvporcupine
import struct
import os
import numpy as np

class WakeWordListener:
    def __init__(self):
        # Grab the key from env variables so I don't hardcode sensitive stuff
        self.ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

        # Locating the custom .ppn model file for "Hello-Vision"
        self.KEYWORD_PATH = os.path.join(
            PROJECT_ROOT,
            "models",
            "Hello-Vision_en_windows_v4_0_0.ppn"
        )

        # Make sure the model file actually exists before trying to load it
        if not os.path.exists(self.KEYWORD_PATH):
            raise FileNotFoundError(f"❌ Error: Could not find '{self.KEYWORD_PATH}'")

        try:
            # Initialize Porcupine with the access key and the custom keyword path
            self.porcupine = pvporcupine.create(
                access_key=self.ACCESS_KEY,
                keyword_paths=[self.KEYWORD_PATH]
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Porcupine: {e}")

    def process_audio(self, pcm_bytes):
        # Trimming the last byte if it's odd to avoid errors during int16 conversion
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:-1]

        # Convert the raw byte stream into 16-bit PCM format that Porcupine understands
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
        
        # Porcupine processes audio in specific chunks (usually 512 samples)
        frame_length = self.porcupine.frame_length
        
        # Loop through the audio buffer frame by frame
        for i in range(0, len(pcm) - frame_length, frame_length):
            frame = pcm[i : i + frame_length]
            result = self.porcupine.process(frame)
            # If result is 0 or higher, it means the wake word was detected
            if result >= 0:
                print("🔥 [WAKEWORD] Match found in audio stream!")
                return True
        return False

    def cleanup(self):
        # Always delete the instance to free up the hardware resources/memory
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()