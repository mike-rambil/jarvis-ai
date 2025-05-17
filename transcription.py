import whisper
import scipy.io.wavfile
import os

def transcribe_audio(audio_data, fs):
    temp_wav = 'temp.wav'
    scipy.io.wavfile.write(temp_wav, fs, audio_data)
    try:
        model = whisper.load_model('base')
        result = model.transcribe(temp_wav)
        return result['text']
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav) 