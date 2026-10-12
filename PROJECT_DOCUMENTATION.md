# Foximize AI Voice & Call Assistant
## Comprehensive Technical Documentation, Architecture Guide & Future Opportunities Roadmap

---

### Executive Overview

The **Foximize AI Voice & Call Assistant** is a production-grade, conversational voice AI system developed for **Foximize Technologies** (Kochi, Kerala). It functions as an intelligent, conversational agent capable of engaging in real-time spoken and textual conversations with prospective students, clients, and partners. 

The assistant provides instant, factual answers regarding Foximize Academy programs (Data Science with Gen AI, Data Analytics with Gen AI, Python Full Stack with Gen AI), fees, registration, placement assistance, and corporate software services.

Engineered with local-first, low-resource hardware efficiency, the entire AI pipeline—including **Speech-to-Text (STT)**, **Retrieval-Augmented Generation (RAG)**, **Large Language Model (LLM) reasoning**, and **Text-to-Speech (TTS) Voice Cloning**—is customized to run efficiently on an **Apple Silicon MacBook Air (M1, 8 GB Unified RAM)** without incurring per-token cloud API bills or external vendor lock-in.

---

## 1. End-to-End System Architecture

The assistant executes an end-to-end, multi-stage AI pipeline across voice and text modalities:

```
                  ┌──────────────────────────────────────────────┐
                  │             User Web Audio Client            │
                  │   (Microphone Capture / HTML5 Audio Stream)  │
                  └───────────────┬──────────────────────────────┘
                                  │ Multipart Audio (WAV/WebM)
                                  ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Backend (app/api.py)                         │
│  - Non-blocking Threadpool Offloading (run_in_threadpool)                      │
│  - Path Traversal & Payload Security Validation                                │
│  - Background Temporary File Cleanups                                         │
└────────┬────────────────────────┬──────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
┌──────────────────┐    ┌────────────────────────────────────────────────────────┐
│  Speech-to-Text  │    │            Retrieval-Augmented Generation (RAG)        │
│   (app/stt.py)   │    │                       (app/rag.py)                     │
│                  │    │                                                        │
│  Faster-Whisper  │    │  1. Vector DB: ChromaDB (Local SQLite Persistent Store)│
│  (Base model)    ├───►│  2. Embedding: sentence-transformers/all-MiniLM-L6-v2  │
│  INT8 quantized  │    │  3. Boundary-aware paragraph/sentence chunker          │
│  4 CPU Threads   │    │  4. Similarity Search: Top-K cosine/L2 match           │
└──────────────────┘    └──────────────────────────┬─────────────────────────────┘
                                                   │
                                                   ▼ Grounded Context + User Query
┌────────────────────────────────────────────────────────────────────────────────┐
│                          LLM Reasoning (app/llm.py)                            │
│                                                                                │
│  - Model: Local Ollama (qwen3:4b)                                              │
│  - Prompt Guardrails: 1-3 concise spoken sentences, strict zero-hallucination   │
│  - Fault Tolerance: HTTP session pooling, exponential retries, offline fallback│
│  - Text Sanitizer: Strips markdown, asterisks, hashes, and bullets for TTS     │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │ Clean Spoken Text
                                       ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│                       Text-to-Speech & Voice Cloning                           │
│                                (app/tts.py)                                    │
│                                                                                │
│  1. Pronunciation Normalizer: ₹/Rs/INR -> words ("fifty-five thousand rupees") │
│  2. Primary Engine: F5-TTS (Few-shot Zero-Shot Voice Cloning via ref audio)    │
│  3. Fallback Engine: Kokoro-MLX (Ultra-fast Apple Silicon Metal GPU synthesis) │
│  4. WAV Stitching: Natural 150ms inter-sentence conversational pauses          │
└──────────────────────────────────────┬─────────────────────────────────────────┘
                                       │ Synthesized WAV
                                       ▼
                  ┌──────────────────────────────────────────────┐
                  │             Client Playback Stream           │
                  │   (Audio Player + Live Transcript + Sources) │
                  └──────────────────────────────────────────────┘
```

---

## 2. Full Project Structure & File Directory

Below is the directory hierarchy of the codebase, detailing the purpose and responsibility of every file:

```
AI CALL ASSISTANT POJECT/
├── data/                                         # Root shared data directory
│   └── chroma/                                   # Root vector database files
│
└── foximize-ai-call-assistant/                   # Core application repository
    ├── .gitignore                                # Prevents audio caches & bytecodes from Git
    ├── README.md                                 # Quick-start instructions & config table
    ├── requirements.txt                          # Pinned Python package dependencies
    │
    ├── app/                                      # Application Source Code
    │   ├── __init__.py                           # Module initializer
    │   ├── api.py                                # FastAPI REST endpoints, security & routes
    │   ├── stt.py                                # Faster-Whisper Speech-to-Text engine
    │   ├── speech.py                             # Compatibility wrapper delegating to stt.py
    │   ├── rag.py                                # ChromaDB vector search & text chunking
    │   ├── index_knowledge.py                    # Knowledge base ingestion CLI script
    │   ├── llm.py                                # Ollama client, prompt rules & retry logic
    │   ├── tts.py                                # F5-TTS cloning + Kokoro-MLX fallback + normalizer
    │   ├── tts_kokoro_backup.py                  # Standalone Kokoro-MLX script
    │   ├── main.py                               # Terminal-based pipeline runner (STT->RAG->LLM)
    │   └── test_speech.py                        # Scratch audio recording test
    │
    ├── frontend/                                 # User Interface
    │   └── index.html                            # Interactive Web Audio microphone & text chat UI
    │
    ├── knowledge_base/                           # Ground Truth Knowledge Corpus (Text Files)
    │   ├── admission.txt                         # Registration steps (₹2000), offer letter, modes
    │   ├── company.txt                           # Foximize background, Kochi offices, services
    │   ├── contact.txt                           # Official phone numbers, email, physical addresses
    │   ├── courses.txt                           # Data Science, Data Analytics, Python Full Stack
    │   ├── faq.txt                               # Frequently asked questions and standard answers
    │   ├── fees.txt                              # Fee schedules (₹55k, ₹45k, ₹40k) & payment plans
    │   ├── internships.txt                       # College group internship guidelines
    │   └── placement.txt                         # 100% placement assurance policy criteria
    │
    ├── data/                                     # Local Runtime Assets & Database
    │   ├── audio_temp/                           # Temporary generated user & response WAV files
    │   ├── chroma/                               # ChromaDB vector store (SQLite & parquet vectors)
    │   ├── voice_reference.wav                   # 24kHz master voice sample used for F5-TTS cloning
    │   ├── test.wav                              # Baseline sample audio file for testing
    │   ├── test_kokoro.wav                       # Sample output synthesized with Kokoro
    │   ├── test_voice.wav                        # Voice sample audio
    │   ├── chunk_test.wav                        # Test chunk audio
    │   ├── cpu_test/                             # Benchmarking artifacts on CPU
    │   ├── mps_test/                             # Benchmarking artifacts on Apple Silicon Metal
    │   └── quality_test/                         # Audio fidelity validation clips
    │
    ├── tests/                                    # Automated Test Suite
    │   ├── run_tests.py                          # Built-in zero-dependency unittest runner (13 tests)
    │   ├── test_api_security.py                  # Path traversal, empty inputs, status codes
    │   ├── test_llm.py                           # Markdown stripping & offline fallback tests
    │   ├── test_rag.py                           # Text chunking & vector search tests
    │   ├── test_stt_proxy.py                     # Singleton proxy & missing audio error tests
    │   └── test_text_processing.py               # Number-to-words & currency conversion tests
    │
    ├── test_stt.py                               # Quick CLI check for STT transcription
    └── test_tts.py                               # Quick CLI check for Kokoro TTS synthesis
```

---

## 3. Technology Stack & Tools Used

### A. Speech-to-Text (STT)
* **faster-whisper (v1.2.1)**: Reimplementation of OpenAI's Whisper model utilizing **CTranslate2**, providing up to 4x faster execution and significantly lower memory overhead than vanilla PyTorch Whisper.
* **Quantization & Threads**: Configured for `int8` CPU quantization with `cpu_threads=4`, matching the 4 performance cores of the Apple M1 processor.
* **Beam Search**: Set to `beam_size=1` for low-latency greedy decoding.

### B. Retrieval-Augmented Generation (RAG) & Vector Database
* **ChromaDB (v1.5.9)**: Lightweight, serverless vector database operating in persistent mode (`PersistentClient`), saving vector embeddings directly to disk with zero server overhead.
* **Sentence-Transformers (`all-MiniLM-L6-v2`)**: Compact embedding model producing 384-dimensional dense vectors with exceptional semantic accuracy and minimal CPU footprint.
* **Custom Boundary-Aware Text Chunker**: Splits knowledge documents preserving paragraph boundaries (`\n\n`), preventing numbers, curriculum details, or phone numbers from splitting across chunks.

### C. Large Language Model (LLM) Inference
* **Ollama (`qwen3:4b`)**: Local language model execution via Ollama. The 4-billion parameter Qwen model provides rapid response generation with strong instruction following and zero cloud latency/cost.
* **HTTP Connection Pooling (`requests.Session` + `urllib3.util.retry.Retry`)**: Reuses socket connections to localhost Ollama and applies automatic retry backoff.
* **Graceful Degradation Fallbacks**: If Ollama is offline or experiences a timeout, the system extracts key sentences directly from the retrieved RAG context and responds reliably without throwing unhandled 500 errors.

### D. Voice Synthesis & Voice Cloning (TTS)
* **F5-TTS (v1.1.22)**: Advanced flow-matching zero-shot voice cloning model capable of cloning a speaker's voice using only a reference audio sample (`voice_reference.wav`) without model retraining.
* **Kokoro-MLX (v0.1.2)**: High-speed, Apple Silicon Metal-accelerated (MLX) TTS serving as an instant fallback or high-speed alternative (`TTS_ENGINE=kokoro`).
* **Text Preprocessor & Indian Currency Normalizer**: Transforms raw symbols into speakable phrases:
  - `₹55,000` $\rightarrow$ *"fifty-five thousand rupees"*
  - `100%` $\rightarrow$ *"one hundred percent"*
  - `8 months` $\rightarrow$ *"eight months"*
* **Sentence Stitcher & Audio Combiner (`wave`, `shutil`)**: Splits long answers into sentence chunks, synthesizes them, and merges them with realistic 150ms inter-sentence conversational silence.

### E. Backend Framework & Web Layer
* **FastAPI (v0.141.1) & Starlette**: Modern, high-performance asynchronous web framework.
* **Uvicorn (v0.53.0)**: Lightning-fast ASGI web server implementation.
* **FastAPI Concurrency (`run_in_threadpool`)**: Crucial architectural design: offloads CPU-intensive inference (Whisper STT, MiniLM Embeddings, F5-TTS) to thread pools, preventing blocking of the async event loop.
* **Pydantic (v2.13.5)**: Strict data validation and schema management.

### F. Frontend
* **Vanilla HTML5 / Modern CSS / JavaScript**: Zero bulky frontend dependencies; renders directly in any modern desktop or mobile browser.
* **Web Audio API (`MediaRecorder`)**: Captures microphone audio streams dynamically and transfers them as `multipart/form-data` blobs to the backend.

### G. Security & Reliability
* **Path Traversal Protection**: Regex sanitization enforcing strict filename patterns (`^[a-zA-Z0-9_\-]+\.wav$`) and path containment checks preventing directory traversal attacks (`/audio/../../etc/passwd`).
* **Payload Size Limits**: Strict 15 MB audio upload limit (`MAX_UPLOAD_SIZE = 15 * 1024 * 1024`).
* **Disk Leak Protection**: Automatic background cleanup tasks deleting temporary audio files older than 30 minutes.

---

## 4. Key Engineering Innovations & Problem-Solving

### 1. Conquering the 8 GB RAM Limit on MacBook Air M1
* **The Problem**: Running an STT model, an embedding model, an LLM, and a neural voice cloner concurrently in 8 GB RAM easily causes system out-of-memory (OOM) crashes and aggressive swap thrashing.
* **The Solution**: 
  1. Utilized **Lazy-Loaded Singletons** (`_LazyProxy` pattern): Models are only initialized into memory on the first request rather than during startup/import.
  2. Unified `speech.py` and `stt.py` to prevent duplicate Whisper model instances in memory.
  3. Deployed `faster-whisper` in `int8` quantization, consuming under 300 MB RAM compared to 1.5+ GB for FP32.
  4. Local LLM running in quantized GGUF format via Ollama.

### 2. Eliminating Event-Loop Freezes
* **The Problem**: In FastAPI, executing heavy synchronous operations (like neural audio synthesis taking 3-5 seconds) inside `async def` routes freezes the entire server, rendering all other users unable to load the page or send text chats.
* **The Solution**: Every heavy ML operation (`transcribe_audio`, `search_knowledge`, `ask_llm`, `text_to_speech`) is executed via `fastapi.concurrency.run_in_threadpool`, ensuring the async event loop remains 100% responsive.

### 3. Speech-Aware Text Post-Processing
* **The Problem**: Raw LLMs generate markdown symbols (`**Data Science**`, `- bullet points`, `### Headings`), and currency symbols (`₹55,000`). When passed directly to TTS, audio synthesizers speak literal punctuation ("*asterisk asterisk Data Science asterisk asterisk*") or mispronounce rupee symbols.
* **The Solution**: 
  1. Custom regex cleaning pipeline (`clean_llm_response`) strips all markdown before passing text to TTS.
  2. Built-in Indian number-to-words engine converts numbers and Indian denominations (lakhs, crores, thousands) into natural conversational English.

---

## 5. Knowledge Base & Business Domain

The assistant is strictly grounded on official **Foximize Technologies** data (`knowledge_base/`):

| Domain | Key Facts Grounded in Assistant |
|---|---|
| **Company Background** | Software development, IT training (Foximize Academy), and HR consultancy based in Kochi, Kerala. |
| **Data Science with Gen AI** | Duration: 8 Months \| Fee: ₹55,000 \| Available: Online, Offline, Hybrid |
| **Data Analytics with Gen AI** | Duration: 6 Months \| Fee: ₹45,000 \| Available: Online, Offline, Hybrid |
| **Python Full Stack with Gen AI**| Duration: 6 Months \| Fee: ₹40,000 \| Available: Online, Offline, Hybrid |
| **Admission Process** | 1. Registration (`₹2,000`) $\rightarrow$ 2. Offer letter issued with joining date $\rightarrow$ 3. Remaining fee paid in full or installments. |
| **Placement Assurance** | 100% placement assurance based on academic performance. |
| **Internships** | Paid internships available specifically for college groups. |
| **Contact Channels** | Phone: +91 6282082651, +91 8848797657 \| Email: info@foximize.com |

---

## 6. Commercial & Technical Opportunities (What You Can Do Next)

This project provides a robust foundation. Below are high-value commercial and technical directions to scale it into an enterprise platform:

### Opportunity 1: Telephony & VoIP Integration (Real Phone Calls)
* **What it enables**: Convert the web assistant into an actual phone number where incoming customer calls are automatically answered by the AI, or outbound calls are initiated to follow up on inquiries.
* **Implementation Plan**:
  - Integrate with **Twilio Voice API**, **Exotel**, or **Plivo** using Webhooks or WebSockets.
  - Connect to an open-source PBX like **Asterisk** or **FreeSWITCH** via SIP trunking.
  - Handle telephony audio codecs (G.711 $\mu$-law/a-law, 8kHz sampling rate) with fast resampling.

### Opportunity 2: Full-Duplex WebRTC Streaming (Ultra-Low Latency < 600ms)
* **What it enables**: Current pipeline processes audio in turns (record audio $\rightarrow$ upload $\rightarrow$ transcribe $\rightarrow$ synthesize $\rightarrow$ download $\rightarrow$ play). Upgrading to streaming allows natural interruptions and instant conversational responses.
* **Implementation Plan**:
  - Replace REST polling with **WebSockets** or **WebRTC** data/audio channels.
  - Implement streaming STT (transcribe as the user speaks).
  - Stream LLM tokens directly to a streaming TTS chunker to start playing audio before the sentence is even fully completed.

### Opportunity 3: Multi-Lingual & Regional Indian Languages
* **What it enables**: Enabling customers across Kerala and India to speak in their native languages (**Malayalam, Hindi, Tamil, Kannada, Telugu**).
* **Implementation Plan**:
  - Deploy **AI4Bharat Indic-Whisper** or **Bhashini API** for Indian accent and regional language speech recognition.
  - Implement regional language TTS voices using **AI4Bharat Indic-TTS**.

### Opportunity 4: Automated CRM & Calendar Appointment Booking
* **What it enables**: Instead of just answering questions, the assistant takes action—booking campus visits, scheduling admission counseling, and creating leads.
* **Implementation Plan**:
  - Implement **Function Calling / Tool Use** in the LLM.
  - Connect with CRM APIs (**HubSpot, Zoho, Salesforce, Google Sheets**).
  - Integrate **Google Calendar / Cal.com** for automatic counseling slot booking.
  - Trigger instant WhatsApp or SMS confirmation messages with registration links.

### Opportunity 5: Outbound Lead Qualification & Follow-Up Engine
* **What it enables**: Proactively call prospective students who filled an inquiry form on social media or the website.
* **Implementation Plan**:
  - Ingestion queue (RabbitMQ / Redis Queue) that triggers automated outbound calls.
  - Dynamic conversation scripts tailored to whether the lead inquired about Data Science or Python Full Stack.
  - Sentiment analysis and call summary generation stored in the sales dashboard.

### Opportunity 6: Enterprise Analytics & Admin Dashboard
* **What it enables**: Providing Foximize management with actionable business intelligence.
* **Implementation Plan**:
  - Real-time dashboard displaying call metrics: average call duration, top asked questions, conversion rates, and sentiment analysis.
  - Quality assurance review with audio playback and transcripts of all past customer interactions.

---

## 7. How to Run & Deploy the Project

### Prerequisites
* Python 3.10 or 3.12+
* [Ollama](https://ollama.ai/) installed and running locally:
  ```bash
  ollama run qwen3:4b
  ```

### Step 1: Install Dependencies
```bash
cd foximize-ai-call-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Ingest the Knowledge Base
```bash
python -m app.index_knowledge
```

### Step 3: Launch the Assistant Server
```bash
uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```

### Step 4: Access the UI
Open your browser and navigate to:
```
http://127.0.0.1:8000
```

### Step 5: Run Automated Tests
```bash
python tests/run_tests.py
```

---

## 8. Summary of API Endpoints

| Method | Endpoint | Description | Sample Request / Body |
|---|---|---|---|
| `GET` | `/` | Serves the web-based voice & text interface | None |
| `POST` | `/chat` | Fast text-based question answering | `{"message": "What is the fee for Data Science?"}` |
| `POST` | `/voice-chat` | Full audio-to-audio voice conversation pipeline | `multipart/form-data; file=@user_voice.wav` |
| `GET` | `/audio/{filename}`| Secure retrieval of synthesized voice audio | `filename=response_abc123.wav` |

---

*Authored by Sumayya | Virtual Assistant & AI Systems Development for Foximize Technologies*
