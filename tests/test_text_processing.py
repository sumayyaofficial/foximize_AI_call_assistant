import pytest
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.tts import prepare_text_for_speech, split_text, _int_to_words


def test_int_to_words():
    assert _int_to_words(0) == "zero"
    assert _int_to_words(5) == "five"
    assert _int_to_words(20) == "twenty"
    assert _int_to_words(45) == "forty-five"
    assert _int_to_words(100) == "one hundred"
    assert _int_to_words(2000) == "two thousand"
    assert _int_to_words(40000) == "forty thousand"
    assert _int_to_words(45000) == "forty-five thousand"
    assert _int_to_words(55000) == "fifty-five thousand"


def test_prepare_text_currency_normalization():
    # Rupee symbols
    text = "The fee is ₹55,000 for 8 months."
    prepared = prepare_text_for_speech(text)
    assert "fifty-five thousand rupees" in prepared
    assert "eight months" in prepared

    text2 = "Registration payment is ₹2000."
    prepared2 = prepare_text_for_speech(text2)
    assert "two thousand rupees" in prepared2

    text3 = "Course fee is Rs. 40,000."
    prepared3 = prepare_text_for_speech(text3)
    assert "forty thousand rupees" in prepared3


def test_prepare_text_markdown_and_percentage():
    text = "**Placement Assurance**: 100% placement assurance."
    prepared = prepare_text_for_speech(text)
    assert "**" not in prepared
    assert "one hundred percent" in prepared


def test_split_text_sentences():
    text = "Hello, welcome to Foximize. Our courses include Data Science and Full Stack. Please visit our website."
    chunks = split_text(text, max_chars=80)
    assert len(chunks) >= 2
    # Ensure all chunks are non-empty strings
    for chunk in chunks:
        assert len(chunk) > 0
        assert len(chunk) <= 100
