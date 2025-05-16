from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import simpleaudio as sa
import numpy as np

if __name__ == "__main__":
    preload_models()
    text = "How are you micheal, hope everything is going great!"
    audio_array = generate_audio(text)
    # Convert to 16-bit PCM for compatibility with simpleaudio
    audio_int16 = np.int16(audio_array / np.max(np.abs(audio_array)) * 32767)
    output_wav = "bark_test_generation.wav"
    write_wav(output_wav, SAMPLE_RATE, audio_int16)
    print(f"Audio generated and saved to {output_wav}.")
    try:
        wave_obj = sa.WaveObject.from_wave_file(output_wav)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"Error playing audio: {e}") 