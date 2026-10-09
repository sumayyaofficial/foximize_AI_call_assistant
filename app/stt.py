from faster_whisper import WhisperModel


# Lightweight model for your M1 8GB Mac
MODEL_SIZE = "base"

model = WhisperModel(
    MODEL_SIZE,
    device="cpu",
    compute_type="int8"
)


def transcribe_audio(audio_path):
    """
    Convert audio file into text.
    """

    segments, info = model.transcribe(
        audio_path,
        beam_size=1
    )

    text = " ".join(
        segment.text.strip()
        for segment in segments
    )

    return text