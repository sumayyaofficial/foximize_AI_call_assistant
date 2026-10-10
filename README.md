# Foximize AI Voice Assistant

An end-to-end voice and text customer assistant for **Foximize Technologies**, optimized for Apple Silicon (MacBook Air M1, 8 GB RAM).

---

## 🏛️ Architecture Overview

The system features an integrated voice-to-voice and text pipeline:
1. **Frontend (`frontend/index.html`)**: Interactive Web Audio interface supporting real-time microphone recording and direct text queries.
2. **FastAPI Server (`app/api.py`)**: Asynchronous, non-blocking API orchestration running CPU/ML tasks in isolated thread pools.
3. **Speech-to-Text (`app/stt.py`)**: Faster-Whisper (`base` model, int8 quantization, 4 CPU threads for M1).
4. **Knowledge Retrieval / RAG (`app/rag.py`)**: ChromaDB persistent vector database with `sentence-transformers/all-MiniLM-L6-v2` and paragraph/sentence-aware chunking.
5. **LLM Reasoning (`app/llm.py`)**: Local Ollama integration (`qwen3:4b`) with automatic retries, timeouts, and natural speech response formatting.
6. **Voice Synthesis (`app/tts.py`)**: F5-TTS voice cloning using `data/voice_reference.wav` with automated fallback and support for Metal-accelerated `kokoro_mlx`.

---

## ⚙️ Configuration (.env / Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Ollama API endpoint |
| `LLM_MODEL` | `qwen3:4b` | Ollama model name |
| `LLM_TIMEOUT` | `30.0` | Timeout in seconds for LLM reasoning |
| `WHISPER_MODEL_SIZE` | `base` | Faster-Whisper model size |
| `TTS_ENGINE` | `f5-tts` | TTS engine (`f5-tts` or `kokoro`) |
| `TTS_REFERENCE_AUDIO` | `data/voice_reference.wav` | Path to voice cloning reference audio |

---

## 🚀 Running the Assistant

### 1. Index Knowledge Base
To index or update company knowledge documents from `knowledge_base/`:
```bash
python -m app.index_knowledge
```

### 2. Start the Server
```bash
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```

Then visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 🧪 Testing

Run automated tests:
```bash
python tests/run_tests.py
```
*(Runs with Python's built-in `unittest` runner, no external test dependencies required).*
