import pvporcupine
import pyaudio
import struct
import os

class WakeWordListener:
    def __init__(self):
        # =========================================
        # CONFIGURATION
        # =========================================
        self.ACCESS_KEY = "nEt7LmdReOz/ThXyOYdyyw99mCsBmd1wm9o1QP/Tw0E/rNzfgSdIGQ=="  
        self.KEYWORD_PATH = "Hello-Vision_en_windows_v4_0_0.ppn"
        # =========================================
        

        if not os.path.exists(self.KEYWORD_PATH):
            raise FileNotFoundError(f"❌ Error: Could not find '{self.KEYWORD_PATH}'")

        try:
            self.porcupine = pvporcupine.create(
                access_key=self.ACCESS_KEY,
                keyword_paths=[self.KEYWORD_PATH]
            )
            
            self.pa = pyaudio.PyAudio()
        
            # Change input_device_index to the ID of your External Mic from the list above
            self.audio_stream = self.pa.open(
                rate=self.porcupine.sample_rate,
                channels=1,
                format=pyaudio.paInt16,
                input=True,
                frames_per_buffer=self.porcupine.frame_length,
                input_device_index=24  # <--- SET THIS TO YOUR EXTERNAL MIC INDEX
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Porcupine: {e}")

    def listen(self):
        """
        Listens continuously until the Wake Word is detected.
        Returns True when it hears 'Hello Vision'.
        """
        try:
            while True:
                pcm = self.audio_stream.read(self.porcupine.frame_length, exception_on_overflow=False)
                pcm = struct.unpack_from("h" * self.porcupine.frame_length, pcm)
                
                keyword_index = self.porcupine.process(pcm)
                
                if keyword_index >= 0:
                    return True # Detected!
                    
        except KeyboardInterrupt:
            return False

    def cleanup(self):
        if hasattr(self, 'porcupine'): self.porcupine.delete()
        if hasattr(self, 'audio_stream'): self.audio_stream.close()
        if hasattr(self, 'pa'): self.pa.terminate()