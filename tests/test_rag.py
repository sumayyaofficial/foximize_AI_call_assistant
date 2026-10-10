import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.rag import chunk_text, search_knowledge


def test_chunk_text_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n\n  ") == []


def test_chunk_text_paragraphs():
    doc = """Foximize Technologies is a software company.
We build AI solutions.

Our Academy provides hands-on training.
Students work on live projects."""
    chunks = chunk_text(doc, chunk_size=200)
    assert len(chunks) >= 1
    # Check that sentences aren't randomly truncated
    combined = " ".join(chunks)
    assert "software company" in combined
    assert "hands-on training" in combined


def test_search_knowledge_empty_query():
    results = search_knowledge("")
    assert results == []
    results_whitespace = search_knowledge("    ")
    assert results_whitespace == []
