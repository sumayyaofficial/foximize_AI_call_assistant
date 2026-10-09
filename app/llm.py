import requests


OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"


def ask_llm(user_message, context=""):

    system_prompt = f"""
You are the AI voice assistant for Foximize Technologies.

Your job is to answer customer questions about Foximize
using the provided company information.

Rules:
- Give short, natural answers.
- Do not invent company information.
- If the information is unavailable, say that you don't have
  that information.
- Your response will be converted into speech, so avoid
  unnecessary formatting.

Company Context:
{context}
"""

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload
    )

    response.raise_for_status()

    return response.json()["message"]["content"]