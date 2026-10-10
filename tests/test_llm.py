import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.llm import clean_llm_response, ask_llm


def test_clean_llm_response_formatting():
    raw_markdown = """### Foximize Courses
Here are the options:
* **Data Science with Gen AI**: 8 months
* *Data Analytics with Gen AI*: 6 months
# Contact us at info@foximize.com"""

    cleaned = clean_llm_response(raw_markdown)
    assert "###" not in cleaned
    assert "**" not in cleaned
    assert "*" not in cleaned
    assert "Data Science with Gen AI" in cleaned
    assert "Data Analytics with Gen AI" in cleaned


def test_ask_llm_offline_fallback():
    # When Ollama is offline or unreachable, ask_llm should return a graceful fallback string
    # instead of crashing with an unhandled exception.
    result = ask_llm("What is the fee for Data Science?", context="Data Science fee is 55000.")
    assert isinstance(result, str)
    assert len(result) > 0
