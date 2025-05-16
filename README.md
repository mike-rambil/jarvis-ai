# Jarvis: Local Voice Agent with LlamaIndex, Whisper, and pyttsx3

## Overview

This project is a local voice agent that:

- Listens for a wake word ("Hey Jarvis") via your microphone
- Transcribes your speech to text **locally** using OpenAI Whisper
- Retrieves context and answers from your own PDF documents using LlamaIndex
- Generates natural language answers with a local Llama model (Hugging Face)
- Speaks the answer back to you using **pyttsx3** (local TTS)

## Features

- **Wake word detection** ("Hey Jarvis")
- **Offline document Q&A** (index your own PDFs)
- **Local LLM support** (no OpenAI API required)
- **Local real-time speech-to-text and text-to-speech**

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

4. **(Optional) Configure TTS parameters:**

   - By default, pyttsx3 uses your system's built-in voices.
   - You can change the voice or rate in `main.py` if you want a different sound.

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
- The agent will answer using your PDF and speak the response.

## Model Downloads

- The first time you run, Whisper and Hugging Face models will be downloaded automatically.
- All models run locally for speech-to-text and text-to-speech (no API key required).
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

See the [LlamaIndex documentation](https://docs.llamaindex.ai/en/stable/) for more on supported file types and advanced features.

## License

This project is for educational and personal use. Check the licenses of all third-party models and APIs you use.

---

**Enjoy your local AI voice agent!**
