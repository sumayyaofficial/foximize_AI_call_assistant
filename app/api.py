import os
import re
import time
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .llm import ask_llm
from .rag import search_knowledge
from .stt import transcribe_audio
from .tts import text_to_speech

BASE_DIR = Path(__file__).resolve().parent.parent
AUDIO_TEMP_DIR = BASE_DIR / "data" / "audio_temp"
AUDIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_SIZE = 15 * 1024 * 1024  # 15 MB limit


app = FastAPI(
    title="Foximize AI Voice Assistant",
    version="0.2.0",
    description="Intelligent AI voice assistant for Foximize Technologies"
)


class ChatRequest(BaseModel):
    message: str


def _cleanup_old_audio_files(max_age_seconds: int = 1800):
    """Clean up generated response files older than 30 minutes to prevent disk leaks."""
    try:
        now = time.time()
        for f in AUDIO_TEMP_DIR.glob("*.wav"):
            if f.is_file() and (now - f.stat().st_mtime) > max_age_seconds:
                try:
                    f.unlink()
                except OSError:
                    pass
    except Exception as e:
        print(f"[Cleanup Warning] Error during audio cleanup: {e}")


@app.get("/")
async def home():
    index_file = BASE_DIR / "frontend" / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found")
    return FileResponse(index_file)


@app.post("/chat")
async def chat(request: ChatRequest):
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    query = request.message.strip()

    # 1. Search Foximize knowledge base (offloaded to threadpool)
    results = await run_in_threadpool(search_knowledge, query, top_k=3)

    # 2. Combine retrieved information
    context = "\n\n".join(
        result["text"]
        for result in results
    )

    # 3. Send question + context to LLM
    response = await run_in_threadpool(ask_llm, query, context)

    return {
        "question": query,
        "response": response,
        "sources": [
            result["source"]
            for result in results
        ]
    }


@app.post("/voice-chat")
async def voice_chat(
    file: Annotated[UploadFile, File()],
    background_tasks: BackgroundTasks
):
    # Enforce background file cleanup
    background_tasks.add_task(_cleanup_old_audio_files)

    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail="Audio file exceeds 15MB limit")

    input_id = uuid.uuid4().hex
    input_path = AUDIO_TEMP_DIR / f"temp_{input_id}.wav"

    try:
        input_path.write_bytes(content)

        # 1. Speech to Text (run non-blocking on threadpool)
        text = await run_in_threadpool(transcribe_audio, str(input_path))
        if not text:
            text = "(inaudible voice input)"

        # 2. RAG Knowledge retrieval
        results = await run_in_threadpool(search_knowledge, text, top_k=3)
        context = "\n\n".join(
            result["text"]
            for result in results
        )

        # 3. LLM Response
        response = await run_in_threadpool(ask_llm, text, context)

        # 4. Text to Speech
        output_filename = f"response_{uuid.uuid4().hex}.wav"
        output_path = AUDIO_TEMP_DIR / output_filename

        await run_in_threadpool(text_to_speech, response, str(output_path))

        return {
            "transcribed_text": text,
            "response": response,
            "audio_url": f"/audio/{output_filename}",
            "sources": [
                result["source"]
                for result in results
            ]
        }

    finally:
        # Guarantee input temporary file is immediately removed
        try:
            if input_path.exists():
                input_path.unlink()
        except OSError:
            pass


@app.get("/audio/{filename}")
async def get_audio(filename: str):
    # Strict validation against path traversal: only alphanumeric, hyphen, underscore + .wav
    if not re.match(r"^[a-zA-Z0-9_\-]+\.wav$", filename):
        raise HTTPException(status_code=400, detail="Invalid audio filename")

    audio_path = (AUDIO_TEMP_DIR / filename).resolve()

    # Prevent escaping AUDIO_TEMP_DIR
    if audio_path.parent != AUDIO_TEMP_DIR.resolve():
        raise HTTPException(status_code=403, detail="Access denied")

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        filename=filename
    )


