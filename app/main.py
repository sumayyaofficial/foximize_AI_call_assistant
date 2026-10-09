from speech import transcribe_audio
from rag import search_knowledge
from llm import ask_llm


audio_file = "../data/test.wav"


# 1. Speech → Text
question = transcribe_audio(audio_file)

print("User:", question)


# 2. Search knowledge
documents = search_knowledge(question)

context = "\n\n".join(documents)


# 3. LLM response
answer = ask_llm(
    question,
    context
)


print("Foximize AI:", answer)