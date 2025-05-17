# Jarvis: Local Voice Agent with LlamaIndex, Whisper, and ElevenLabs TTS

## Overview

This project is a local voice agent that:

- Listens for a wake word ("Hey Jarvis") via your microphone
- Transcribes your speech to text **locally** using OpenAI Whisper
- Retrieves context and answers from your own PDF documents using LlamaIndex
- Generates natural language answers with a local Llama model (Hugging Face)
- Speaks the answer back to you using **ElevenLabs TTS** (cloud API)

**Want even better accuracy or more natural voices?**

- You can optionally use cloud AI services like **AssemblyAI** (for speech-to-text), **Cartesia** (for TTS), or **OpenAI** (for LLMs) by uncommenting and configuring the relevant code and settings in this project.

## Features

- **Wake word detection** ("Hey Jarvis")
- **Offline document Q&A** (index your own PDFs)
- **Local LLM support** (no OpenAI API required)
- **Local real-time speech-to-text**
- **Cloud TTS with ElevenLabs**
- **All AI-generated audio files are saved in the `ai-content/` folder**
- **Cleanup prompt:** On exit, you can choose to delete all AI-generated audio files

## Setup

1. **Clone the repository**
2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   .\venv\Scripts\Activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure ElevenLabs API:**

   - Create a `.env` file in the project root with:

     ```
     ELEVENLABS_API_KEY=your_elevenlabs_api_key
     ELEVENLABS_VOICE_ID=your_elevenlabs_voice_id
     ```

   - You can find your API key and voice ID in your ElevenLabs dashboard.

5. **Prepare your document:**
   - Place exactly one PDF file in the `docs/` folder. (Only one PDF is supported at a time.)

## Index Your PDF

Run the following command to index your document:

```bash
python index_pdf.py
```

This will create the `environment_index/` directory for fast Q&A.

## Run the Voice Agent

Start the agent with:

```bash
python main.py
```

- The agent will listen for "Hey Jarvis".
- After the wake word, speak your command/question.
- The agent will answer using your PDF and speak the response using ElevenLabs.
- **All generated audio files are saved in the `ai-content/` folder as `tts_<timestamp>.mp3`.**
- **On exit (Ctrl+C), if there are files in `ai-content/`, you will be prompted to delete them.**

## Model Downloads

- The first time you run, Whisper and Hugging Face models will be downloaded automatically.
- All models run locally for speech-to-text and LLM (no OpenAI API required for LLM or STT).
- ElevenLabs TTS requires an internet connection and API key.
- You may need to log in to Hugging Face for gated models.

## Next Level: Global Voice Agent with LiveKit

To take this project to the next level:

- Integrate [LiveKit](https://livekit.io/) to build a web UI for voice input.
- Users can access your agent from anywhere in the world via the internet.
- LiveKit provides real-time audio/video streaming and UI components for web/mobile.
- Your backend can process remote audio just like local audio.

**LiveKit enables multi-user, remote, and browser-based voice agents.**

## Extensibility

This project is designed to be easily extended to fit your needs:

- Read and index **multiple documents** at once.
- Support **different file types** (PDF, DOCX, TXT, Markdown, HTML, etc.) using LlamaIndex's flexible document loaders.
- Add more advanced **search, summarization, or Q&A** features.
- Integrate with other local or cloud LLMs, TTS, or STT engines.
- Build a web or mobile UI for remote access.
- **Use cloud AI services:**
  - Swap in [**AssemblyAI**](https://www.assemblyai.com/) for speech-to-text, [**Cartesia**](https://cartesia.ai/) for TTS (including the ability to [clone your own voice](https://cartesia.ai/)), or [**OpenAI**](https://platform.openai.com/) for LLMs for improved accuracy and naturalness.
  - See the commented configuration/code for AssemblyAI and Cartesia in `main.py` for easy switching.

See the [LlamaIndex documentation](https://docs.llamaindex.ai/en/stable/) for more on supported file types and advanced features.

## License

This project is for educational and personal use. Check the licenses of all third-party models and APIs you use.

---

**Enjoy your local AI voice agent!**
