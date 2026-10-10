"""
Standard unittest test runner for environments without pytest.
Runs all unit and integration test functions across tests/ directory.
"""
import sys
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TextProcessingTests(unittest.TestCase):
    def test_int_to_words(self):
        from app.tts import _int_to_words
        self.assertEqual(_int_to_words(0), "zero")
        self.assertEqual(_int_to_words(5), "five")
        self.assertEqual(_int_to_words(20), "twenty")
        self.assertEqual(_int_to_words(45), "forty-five")
        self.assertEqual(_int_to_words(100), "one hundred")
        self.assertEqual(_int_to_words(2000), "two thousand")
        self.assertEqual(_int_to_words(40000), "forty thousand")
        self.assertEqual(_int_to_words(45000), "forty-five thousand")
        self.assertEqual(_int_to_words(55000), "fifty-five thousand")

    def test_prepare_text_currency(self):
        from app.tts import prepare_text_for_speech
        text = "The fee is ₹55,000 for 8 months."
        prepared = prepare_text_for_speech(text)
        self.assertIn("fifty-five thousand rupees", prepared)
        self.assertIn("eight months", prepared)

        text2 = "Registration payment is ₹2000."
        prepared2 = prepare_text_for_speech(text2)
        self.assertIn("two thousand rupees", prepared2)

        text3 = "Course fee is Rs. 40,000."
        prepared3 = prepare_text_for_speech(text3)
        self.assertIn("forty thousand rupees", prepared3)

    def test_prepare_text_markdown_and_percentage(self):
        from app.tts import prepare_text_for_speech
        text = "**Placement Assurance**: 100% placement assurance."
        prepared = prepare_text_for_speech(text)
        self.assertNotIn("**", prepared)
        self.assertIn("one hundred percent", prepared)

    def test_split_text_sentences(self):
        from app.tts import split_text
        text = "Hello, welcome to Foximize. Our courses include Data Science and Full Stack. Please visit our website."
        chunks = split_text(text, max_chars=80)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertGreater(len(chunk), 0)


class RAGTests(unittest.TestCase):
    def test_chunk_text_empty(self):
        from app.rag import chunk_text
        self.assertEqual(chunk_text(""), [])
        self.assertEqual(chunk_text("   \n\n  "), [])

    def test_chunk_text_paragraphs(self):
        from app.rag import chunk_text
        doc = """Foximize Technologies is a software company.
We build AI solutions.

Our Academy provides hands-on training.
Students work on live projects."""
        chunks = chunk_text(doc, chunk_size=200)
        self.assertGreaterEqual(len(chunks), 1)
        combined = " ".join(chunks)
        self.assertIn("software company", combined)
        self.assertIn("hands-on training", combined)

    def test_search_knowledge_empty_query(self):
        from app.rag import search_knowledge
        self.assertEqual(search_knowledge(""), [])
        self.assertEqual(search_knowledge("    "), [])


class LLMTests(unittest.TestCase):
    def test_clean_llm_response(self):
        from app.llm import clean_llm_response
        raw = """### Foximize Courses
Here are the options:
* **Data Science with Gen AI**: 8 months
* *Data Analytics with Gen AI*: 6 months"""
        cleaned = clean_llm_response(raw)
        self.assertNotIn("###", cleaned)
        self.assertNotIn("**", cleaned)
        self.assertNotIn("*", cleaned)
        self.assertIn("Data Science with Gen AI", cleaned)

    def test_ask_llm_offline_fallback(self):
        from app.llm import ask_llm
        result = ask_llm("What is the fee for Data Science?", context="Data Science fee is 55000.")
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class STTTests(unittest.TestCase):
    def test_stt_missing_file(self):
        from app.stt import transcribe_audio
        with self.assertRaises(FileNotFoundError):
            transcribe_audio("non_existent_audio_path_xyz.wav")

    def test_speech_delegation(self):
        from app.speech import transcribe_audio
        with self.assertRaises(FileNotFoundError):
            transcribe_audio("non_existent_audio_path_xyz.wav")


class APISecurityTests(unittest.TestCase):
    def test_audio_path_traversal_validation_logic(self):
        import re
        from pathlib import Path

        def is_safe_audio_filename(filename: str, base_dir: Path) -> int:
            if not re.match(r"^[a-zA-Z0-9_\-]+\.wav$", filename):
                return 400  # Invalid format / traversal pattern
            audio_path = (base_dir / filename).resolve()
            if audio_path.parent != base_dir.resolve():
                return 403  # Escapes temp dir
            if not audio_path.exists():
                return 404  # Not found
            return 200

        temp_dir = PROJECT_ROOT / "data" / "audio_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Attacks
        self.assertEqual(is_safe_audio_filename("../../requirements.txt", temp_dir), 400)
        self.assertEqual(is_safe_audio_filename("../data/test.wav", temp_dir), 400)
        self.assertEqual(is_safe_audio_filename("response_missing.wav", temp_dir), 404)
        self.assertEqual(is_safe_audio_filename("malicious.py", temp_dir), 400)

        # Valid file
        valid_file = temp_dir / "test_probe.wav"
        valid_file.write_bytes(b"RIFF dummy")
        try:
            self.assertEqual(is_safe_audio_filename("test_probe.wav", temp_dir), 200)
        finally:
            if valid_file.exists():
                valid_file.unlink()

    def test_audio_endpoints_if_fastapi_available(self):
        try:
            import asyncio
            from fastapi import HTTPException
            from app.api import get_audio, chat, ChatRequest
        except Exception as e:
            self.skipTest(f"FastAPI not loadable in current Python runtime: {e}")

        # Path traversal
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(get_audio("../../requirements.txt"))
        self.assertEqual(ctx.exception.status_code, 400)

        # Missing file 404
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(get_audio("response_missing_file_12345.wav"))
        self.assertEqual(ctx.exception.status_code, 404)

        # Empty chat message 400
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(chat(ChatRequest(message="   ")))
        self.assertEqual(ctx.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
