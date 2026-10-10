"""
Speech-to-text module.
Delegates to app.stt to avoid duplicate Whisper models in memory.
"""
from pathlib import Path

try:
    from app.stt import transcribe_audio, get_whisper_model, model
except ImportError:
    from stt import transcribe_audio, get_whisper_model, model

__all__ = ["transcribe_audio", "model", "get_whisper_model"]