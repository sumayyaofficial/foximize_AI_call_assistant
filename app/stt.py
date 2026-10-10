import os
from pathlib import Path


# Lightweight model optimized for M1 8GB Mac
MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")
_model = None


def get_whisper_model():
    """
    Lazy-load the Whisper model singleton to avoid RAM waste at import time.
    """
    global _model
    if _model is None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster_whisper is not installed. Please activate the virtual environment."
            ) from e

        _model = WhisperModel(
            MODEL_SIZE,
            device="cpu",
            compute_type="int8",
            cpu_threads=4  # Optimized for M1 4-performance-core allocation
        )
    return _model


# Maintain backward compatibility for modules accessing stt.model directly
class _ModelProxy:
    def __getattr__(self, name):
        return getattr(get_whisper_model(), name)

    def transcribe(self, *args, **kwargs):
        return get_whisper_model().transcribe(*args, **kwargs)

model = _ModelProxy()


def transcribe_audio(audio_path):
    """
    Convert audio file into text.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    whisper_instance = get_whisper_model()
    segments, info = whisper_instance.transcribe(
        str(path),
        beam_size=1
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    return text.strip()