import numpy as np
import noisereduce as nr

def clean_audio_data(audio_sr_obj):
    """
    Takes a SpeechRecognition AudioData object, 
    removes noise, and returns a new AudioData object.
    """
    # 1. Convert SpeechRecognition object to raw numpy array
    # We get the raw bytes and convert to int16 (standard for wav)
    raw_data = audio_sr_obj.get_raw_data(convert_rate=16000, convert_width=2)
    audio_np = np.frombuffer(raw_data, dtype=np.int16)

    # 2. Apply Noise Reduction
    # stationary=True helps with constant background noise like fans
    reduced_noise = nr.reduce_noise(y=audio_np, sr=16000, stationary=True)

    # 3. Convert back to bytes
    clean_bytes = reduced_noise.tobytes()

    return clean_bytes