import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from app.stt import transcribe_audio
    from app.rag import search_knowledge
    from app.llm import ask_llm
except ImportError:
    from stt import transcribe_audio
    from rag import search_knowledge
    from llm import ask_llm


audio_file = BASE_DIR / "data" / "test.wav"


# 1. Speech → Text
if audio_file.exists():
    question = transcribe_audio(str(audio_file))
else:
    question = "What courses does Foximize offer?"
    print(f"Warning: {audio_file} not found, using default question.")

print("User:", question)


# 2. Search knowledge
documents = search_knowledge(question)

context = "\n\n".join(
    doc["text"] if isinstance(doc, dict) else str(doc)
    for doc in documents
)


# 3. LLM response
answer = ask_llm(
    question,
    context
)


print("Foximize AI:", answer)