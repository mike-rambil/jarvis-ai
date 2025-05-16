import os
from dotenv import load_dotenv
load_dotenv()
import time
# Audio capture
import sounddevice as sd
import numpy as np
# AssemblyAI
import requests
# LlamaIndex
# (Assume llama_index is installed and docs are indexed)
# Cartesia (Assume API usage via requests)
import whisper
import pyttsx3

# LlamaIndex imports
from llama_index.core import  StorageContext, load_index_from_storage
from local_llama_qa import load_llama_1b_instruct, ask_llama
from llama_index.core.settings import Settings

# --- CONFIG ---

DOCS_PATH = 'docs/'  # Directory containing your documents
INDEX_DIR = 'environment_index'  # Directory containing your prebuilt index

ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI_API_KEY', 'your-assemblyai-key')
CARTESIA_API_KEY = os.getenv('CARTESIA_API_KEY', 'your-cartesia-key')

# Cartesia AI parameters
# CARTESIA_MODEL_ID = os.getenv('CARTESIA_MODEL_ID', 'sonic-2')
# CARTESIA_VOICE_ID = os.getenv('CARTESIA_VOICE_ID', '1cbda053-e128-48a5-890c-e1d19c99ccbc')
# CARTESIA_SAMPLE_RATE = int(os.getenv('CARTESIA_SAMPLE_RATE', '44100'))
# CARTESIA_ENCODING = os.getenv('CARTESIA_ENCODING', 'pcm_f32le')
# CARTESIA_CONTAINER = os.getenv('CARTESIA_CONTAINER', 'wav')
# CARTESIA_LANGUAGE = os.getenv('CARTESIA_LANGUAGE', 'en')

# --- AUDIO CAPTURE ---
def record_audio(duration=5, fs=16000):
    print(f"Recording {duration}s of audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return audio.flatten(), fs

# --- ASSEMBLYAI TRANSCRIPTION ---
def transcribe_audio(audio_data, fs):
    import scipy.io.wavfile
    temp_wav = 'temp.wav'
    scipy.io.wavfile.write(temp_wav, fs, audio_data)
    model = whisper.load_model('base')
    result = model.transcribe(temp_wav)
    return result['text']

# --- LLAMAINDEX QUERY ---
def load_llamaindex():
    # Load the prebuilt index from disk
    print(f"Loading index from {INDEX_DIR}...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    index = load_index_from_storage(storage_context)
    return index

def query_llamaindex(index, question):
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    return str(response)

# --- CARTESIA SPEECH SYNTHESIS ---
def speak_with_cartesia(text):
    # Use pyttsx3 for local speech synthesis
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error with pyttsx3 TTS: {e}")

# --- MAIN LOOP ---
def main():
    print("Loading LlamaIndex...")
    index = load_llamaindex()
    print("Loading Hugging Face Llama generator...")
    generator = load_llama_1b_instruct()
    print("Ready for wake word detection.")
    while True:
        print("[Listening for wake word: 'Hey Jarvis']")
        audio, fs = record_audio(duration=3)
        transcript = transcribe_audio(audio, fs)
        print(f"[Wake word transcript]: {transcript}")
        if "hey jarvis" in transcript.lower():
            print("Wake word detected! Listening for command...")
            audio, fs = record_audio(duration=5)
            command = transcribe_audio(audio, fs)
            print(f"[User command]: {command}")
            context = query_llamaindex(index, command)
            print(f"LlamaIndex context: {context}")
            answer = ask_llama(command, context=context, generator=generator)
            print(f"Model answer: {answer}")
            speak_with_cartesia(answer)
            print("---")
        else:
            print("Wake word not detected. Continuing to listen...")
        time.sleep(1)

if __name__ == '__main__':
    # Set the local embedding model
    Settings.embed_model = "local:sentence-transformers/all-MiniLM-L6-v2"
    Settings.llm = None
    main() 