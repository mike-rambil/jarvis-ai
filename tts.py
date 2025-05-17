import os
import time
from elevenlabs.client import ElevenLabs
from elevenlabs import play

AI_CONTENT_DIR = "ai-content"
os.makedirs(AI_CONTENT_DIR, exist_ok=True)

ELEVENLABS_API_KEY = os.getenv('ELEVENLABS_API_KEY')
ELEVENLABS_VOICE_ID = os.getenv('ELEVENLABS_VOICE_ID')

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
    # Save audio to file with timestamp in ai-content folder
    timestamp = int(time.time())
    output_file = os.path.join(AI_CONTENT_DIR, f"tts_{timestamp}.mp3")
    with open(output_file, "wb") as f:
        f.write(audio_bytes)
    print(f"Audio saved to {output_file}")
    try:
        play(audio_bytes)
    except Exception as e:
        print(f"Error playing audio: {e}") 