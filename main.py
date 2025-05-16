import os
from dotenv import load_dotenv
load_dotenv()
import time
# Audio capture
import sounddevice as sd
import numpy as np
import whisper
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import simpleaudio as sa

# LlamaIndex imports
from llama_index.core import  StorageContext, load_index_from_storage
from local_llama_qa import load_llama_1b_instruct, ask_llama
from llama_index.core.settings import Settings

# --- CONFIG ---

DOCS_PATH = 'docs/'  # Directory containing your documents
INDEX_DIR = 'environment_index'  # Directory containing your prebuilt index

# --- AUDIO CAPTURE ---
def record_audio(duration=5, fs=16000):
    print(f"Recording {duration}s of audio...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    return audio.flatten(), fs

# --- ASSEMBLYAI TRANSCRIPTION ---
def transcribe_audio(audio_data, fs):
    import scipy.io.wavfile
    import os
    temp_wav = 'temp.wav'
    scipy.io.wavfile.write(temp_wav, fs, audio_data)
    try:
        model = whisper.load_model('base')
        result = model.transcribe(temp_wav)
        return result['text']
    finally:
        if os.path.exists(temp_wav):
            os.remove(temp_wav)

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

def speak_with_bark(text):
    # Preload Bark models (only needs to be done once)
    preload_models()
    # Generate audio from text
    audio_array = generate_audio(text)
    # Save audio to disk
    output_wav = "bark_generation.wav"
    write_wav(output_wav, SAMPLE_RATE, audio_array)
    # Play audio
    try:
        wave_obj = sa.WaveObject.from_wave_file(output_wav)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"Error playing audio: {e}")

# --- MAIN LOOP ---
def main():
    # Check if index exists, if not, build it
    index_file = os.path.join(INDEX_DIR, 'docstore.json')
    if not os.path.exists(index_file):
        print(f"Index file {index_file} not found. Building index...")
        import index_pdf  # This will run the indexing code
    print("Loading LlamaIndex...")
    index = load_llamaindex()
    print("Loading Hugging Face Llama generator...")
    generator = load_llama_1b_instruct()
    print("Ready for wake word detection.")
    try:
        while True:
            print("[Listening for wake word: 'Hey Jarvis']")
            audio, fs = record_audio(duration=3)
            transcript = transcribe_audio(audio, fs)
            print(f"[Wake word transcrip ‼️👀]: {transcript}")
            if "hey jarvis" in transcript.lower():
                print("Wake word detected! Listening for command...")
                audio, fs = record_audio(duration=5)
                command = transcribe_audio(audio, fs)
                print(f"[User command]: {command}")
                context = query_llamaindex(index, command)
                print(f"LlamaIndex context: {context}")
                answer = ask_llama(command, context=context, generator=generator)
                print(f"Model answer: {answer}")
                speak_with_bark(answer)
                print("---")
            else:
                print("Wake word not detected. Continuing to listen...")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting gracefully. Goodbye!")

if __name__ == '__main__':
    # Set the local embedding model
    Settings.embed_model = "local:sentence-transformers/all-MiniLM-L6-v2"
    Settings.llm = None
    main() 