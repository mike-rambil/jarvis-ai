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
from scipy.io.wavfile import write as write_wav
from elevenlabs.client import ElevenLabs
from elevenlabs import play

# LlamaIndex imports
from llama_index.core import  StorageContext, load_index_from_storage
from local_llama_qa import load_llama_1b_instruct, ask_llama
from llama_index.core.settings import Settings

# --- CONFIG ---

DOCS_PATH = 'docs/'  # Directory containing your documents
INDEX_DIR = 'environment_index'  # Directory containing your prebuilt index

ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID')

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

# --- ELEVENLABS SPEECH SYNTHESIS ---
def speak_with_elevenlabs(text):
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        print("ElevenLabs API key or voice ID not set in environment variables.")
        return
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    audio = client.text_to_speech.convert(
        text=text,
        voice_id=ELEVENLABS_VOICE_ID,
        model_id="eleven_turbo_v2_5",
        output_format="mp3_44100_128"
    )
    audio_bytes = b"".join(chunk for chunk in audio if isinstance(chunk, (bytes, bytearray)))
    # Save audio to file with timestamp
    timestamp = int(time.time())
    output_file = f"tts_{timestamp}.mp3"
    with open(output_file, "wb") as f:
        f.write(audio_bytes)
    print(f"Audio saved to {output_file}")
    try:
        play(audio_bytes)
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
                answer = ask_llama(
                    "You are a helpful AI assistant. Respond conversationally and politely.\n\n" + context + "\n\nUser: " + command + "\nAssistant:",
                    generator=generator
                )
                # Only echo the lines that the assistant speaks
                if "Assistant:" in answer:
                    assistant_reply = answer.split("Assistant:")[-1].strip()
                else:
                    assistant_reply = answer.strip()
                # Remove any '<|assistant|>' tokens
                assistant_reply = assistant_reply.replace('<|assistant|>', '').strip()
                print(f"AI🔥: {assistant_reply}")
                speak_with_elevenlabs(assistant_reply)
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