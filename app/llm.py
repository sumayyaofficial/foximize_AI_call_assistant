import os
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
MODEL = os.getenv("LLM_MODEL", "qwen3:4b")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30.0"))

_session = None


def get_http_session():
    """
    Reusable HTTP session with connection pooling and fast retries.
    """
    global _session
    if _session is None:
        _session = requests.Session()
        retries = Retry(
            total=2,
            backoff_factor=0.3,
            status_forcelist=[500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retries, pool_connections=5, pool_maxsize=10)
        _session.mount("http://", adapter)
        _session.mount("https://", adapter)
    return _session


def clean_llm_response(text: str) -> str:
    """
    Strip markdown formatting (asterisks, hashtags, bullets) so TTS reads naturally.
    """
    if not text:
        return ""
    # Remove markdown headers (### )
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Remove bold / italic markdown (**text** or *text*)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)
    text = re.sub(r"_([^_]+)_", r"\1", text)
    # Replace markdown bullet points with a plain sentence structure
    text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.MULTILINE)
    # Collapse multiple whitespace/newlines
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def ask_llm(user_message, context=""):
    """
    Query the local Ollama LLM with user query and RAG context.
    Includes timeout safety and graceful offline fallbacks.
    """
    system_prompt = f"""You are the AI voice assistant for Foximize Technologies.

Your job is to answer customer questions about Foximize using the provided company information.

Rules:
- Give short, natural answers (1 to 3 sentences).
- Do not invent company information.
- If the information is unavailable, say that you don't have that information.
- Your response will be converted directly into speech, so do not use markdown, bullets, or asterisks.

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

    session = get_http_session()

    try:
        response = session.post(
            OLLAMA_URL,
            json=payload,
            timeout=LLM_TIMEOUT
        )
        response.raise_for_status()
        raw_content = response.json()["message"]["content"]
        return clean_llm_response(raw_content)

    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
        print(f"[LLM Warning] Ollama unavailable at {OLLAMA_URL}: {e}")
        # If context is available, formulate a clean fallback based on retrieved context
        if context:
            first_line = context.split("\n")[0].strip()
            return f"Based on Foximize records: {first_line}. For more details, please contact Foximize Technologies directly."
        return "I am currently unable to reach the AI model server. Please make sure Ollama is running or contact Foximize staff directly."

    except Exception as e:
        print(f"[LLM Error] Unexpected error calling Ollama: {e}")
        return "Sorry, I encountered an error while generating a response. Please try again."