import os
import time
from dotenv import load_dotenv
load_dotenv()
from audio import record_audio
from transcription import transcribe_audio
from tts import speak_with_elevenlabs
from llama_index_utils import load_llamaindex, query_llamaindex
from local_llama_qa import load_llama_1b_instruct, ask_llama
from llama_index.core.settings import Settings

INDEX_DIR = 'environment_index'

def create_index():
    index_file = os.path.join(INDEX_DIR, 'docstore.json')
    if not os.path.exists(index_file):
        print(f"Index file {index_file} not found. Building index...")
        import index_pdf  # This will run the indexing code

# --- MAIN LOOP ---
def main():
    create_index()
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
        # Check for files in ai-content and prompt for deletion
        AI_CONTENT_DIR = "ai-content"
        ai_files = [f for f in os.listdir(AI_CONTENT_DIR) if f.startswith('tts_')]
        if ai_files:
            resp = input(f"Delete all AI-generated contents in '{AI_CONTENT_DIR}'? Unless you like to keep them (Y/n): ").strip().lower()
            if resp == 'y':
                for f in ai_files:
                    try:
                        os.remove(os.path.join(AI_CONTENT_DIR, f))
                    except Exception as e:
                        print(f"Could not delete {f}: {e}")
                print(f"All AI-generated contents in '{AI_CONTENT_DIR}' deleted.")

if __name__ == '__main__':
    Settings.embed_model = "local:sentence-transformers/all-MiniLM-L6-v2"
    Settings.llm = None
    main() 