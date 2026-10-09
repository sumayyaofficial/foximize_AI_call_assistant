from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from .llm import ask_llm
from .rag import search_knowledge
from .stt import transcribe_audio
from fastapi.responses import FileResponse
from pathlib import Path
from .tts import text_to_speech
import os
import uuid
from typing import Annotated
from fastapi.staticfiles import StaticFiles


app = FastAPI(
    title="Foximize AI Voice Assistant",
    version="0.1.0"
)


class ChatRequest(BaseModel):
    message: str


from fastapi.responses import FileResponse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

@app.get("/")
async def home():
    return FileResponse(
        BASE_DIR / "frontend" / "index.html"
    )

@app.post("/chat")
def chat(request: ChatRequest):

    # 1. Search Foximize knowledge base
    results = search_knowledge(
        request.message,
        top_k=3
    )

    # 2. Combine retrieved information
    context = "\n\n".join(
        result["text"]
        for result in results
    )

    # 3. Send question + context to Qwen
    response = ask_llm(
        request.message,
        context
    )

    return {
        "question": request.message,
        "response": response,
        "sources": [
            result["source"]
            for result in results
        ]
    }

@app.post("/voice-chat")
async def voice_chat(
    file: Annotated[UploadFile, File()]
):
    input_filename = f"temp_{uuid.uuid4()}.wav"

    with open(input_filename, "wb") as buffer:
        buffer.write(await file.read())

    # Speech to Text
    text = transcribe_audio(input_filename)

    # RAG
    results = search_knowledge(text, top_k=3)

    context = "\n\n".join(
        result["text"]
        for result in results
    )

    # LLM
    response = ask_llm(
        text,
        context
    )

    # Text to Speech
    output_filename = f"response_{uuid.uuid4()}.wav"

    text_to_speech(
        response,
        output_filename
    )

    return {
        "transcribed_text": text,
        "response": response,
        "audio_url": f"/audio/{output_filename}",
        "sources": [
            result["source"]
            for result in results
        ]
    }
@app.get("/audio/{filename}")
async def get_audio(filename: str):
    audio_path = Path(filename)

    if not audio_path.exists():
        return {"error": "Audio file not found"}

    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        filename=filename
    )


