import pvporcupine              # Wake-word detection engine (offline)
import pyaudio                 # Microphone audio streaming
import struct                  # Convert raw audio bytes to integers
import os                      # File existence checks

class WakeWordListener:
    def __init__(self):
        # =========================================
        # CONFIGURATION
        # =========================================
        # Picovoice access key (required for Porcupine)
        self.ACCESS_KEY = "I7DwIm/orJID5Zz3kSPBeYRrt0Ic/yWnZ2eo7rLJUz9hXgklqgpxXw=="  
        
        # Custom wake-word model file ("Hello Vision")
        self.KEYWORD_PATH = "Hello-Vision_en_windows_v4_0_0.ppn"
        # =========================================

        # Ensure wake-word model file exists
        if not os.path.exists(self.KEYWORD_PATH):
            raise FileNotFoundError(f"❌ Error: Could not find '{self.KEYWORD_PATH}'")

        try:
            # Initialize Porcupine with custom wake word
            self.porcupine = pvporcupine.create(
                access_key=self.ACCESS_KEY,
                keyword_paths=[self.KEYWORD_PATH]
            )

            # Initialize microphone audio stream
            self.pa = pyaudio.PyAudio()
            self.audio_stream = self.pa.open(
                rate=self.porcupine.sample_rate,     # Required sample rate
                channels=1,                          # Mono audio
                format=pyaudio.paInt16,              # 16-bit PCM
                input=True,
                frames_per_buffer=self.porcupine.frame_length
            )
        except Exception as e:
            # Catch initialization errors (mic, key, model issues)
            raise RuntimeError(f"Failed to initialize Porcupine: {e}")

    def listen(self):
        """
        Continuously listens to microphone input.
        Returns True when the wake word ('Hello Vision') is detected.
        """
        try:
            while True:
                # Read raw audio frame from microphone
                pcm = self.audio_stream.read(
                    self.porcupine.frame_length,
                    exception_on_overflow=False
                )

                # Convert byte data to signed 16-bit integers
                pcm = struct.unpack_from(
                    "h" * self.porcupine.frame_length,
                    pcm
                )

                # Run wake-word detection
                keyword_index = self.porcupine.process(pcm)

                # Wake word detected
                if keyword_index >= 0:
                    return True

        except KeyboardInterrupt:
            # Allow clean exit on Ctrl+C
            return False

    def cleanup(self):
        # Release system resources cleanly
        if hasattr(self, 'porcupine'):
            self.porcupine.delete()
        if hasattr(self, 'audio_stream'):
            self.audio_stream.close()
        if hasattr(self, 'pa'):
            self.pa.terminate()
