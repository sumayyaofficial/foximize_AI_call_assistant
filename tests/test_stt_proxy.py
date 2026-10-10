import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.stt import transcribe_audio as stt_transcribe
from app.speech import transcribe_audio as speech_transcribe


def test_stt_missing_file_raises_not_found():
    with pytest.raises(FileNotFoundError):
        stt_transcribe("non_existent_audio_path_xyz.wav")


def test_speech_delegates_to_stt():
    # Both stt and speech transcribe should exhibit identical behavior without duplicate models
    with pytest.raises(FileNotFoundError):
        speech_transcribe("non_existent_audio_path_xyz.wav")
