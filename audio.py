import sounddevice as sd
import numpy as np

def record_audio(duration=5, fs=16000):
    print(f"Recording {duration}s of audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return audio.flatten(), fs 