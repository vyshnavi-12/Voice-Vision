# Evaluation/noise_filtering.py

import librosa
import noisereduce as nr
import soundfile as sf
import tempfile


def clean_audio_for_stt(input_path):
    """
    Broadcast-style spectral noise reduction.
    Used only during noisy/unclear evaluation.
    """

    # Load audio @ Whisper expected rate
    y, sr = librosa.load(input_path, sr=16000)

    # First 0.5 sec assumed as background noise profile
    noise_sample = y[0:int(0.5 * sr)]

    reduced = nr.reduce_noise(
        y=y,
        sr=sr,
        y_noise=noise_sample,
        stationary=False,
        prop_decrease=1.0
    )

    # Save cleaned temp file
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, reduced, sr)

    return tmp.name