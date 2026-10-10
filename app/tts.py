import os
import re
import shutil
import subprocess
import uuid
import wave
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

REFERENCE_AUDIO = Path(
    os.getenv("TTS_REFERENCE_AUDIO", str(BASE_DIR / "data" / "voice_reference.wav"))
)

REFERENCE_TEXT = os.getenv(
    "TTS_REFERENCE_TEXT",
    "Hello, welcome to Foximize Technologies. My name is Sumayya, and I’m your virtual assistant. I’m here to help you with information about our courses, training programs, admission process, fees, internships, and other services."
).strip()

TTS_ENGINE = os.getenv("TTS_ENGINE", "f5-tts").lower()
_kokoro_instance = None


def _int_to_words(n: int) -> str:
    """Pure-Python integer to words for natural pronunciation."""
    ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
            'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
            'seventeen', 'eighteen', 'nineteen']
    tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']

    if n == 0:
        return 'zero'
    words = []
    if n >= 10000000:
        words.append(_int_to_words(n // 10000000) + ' crore')
        n %= 10000000
    if n >= 100000:
        words.append(_int_to_words(n // 100000) + ' lakh')
        n %= 100000
    if n >= 1000:
        words.append(_int_to_words(n // 1000) + ' thousand')
        n %= 1000
    if n >= 100:
        words.append(ones[n // 100] + ' hundred')
        n %= 100
    if n >= 20:
        t = tens[n // 10]
        rem = ones[n % 10]
        words.append(f'{t}-{rem}' if rem else t)
    elif n > 0:
        words.append(ones[n])
    return ' '.join(words).strip()


def prepare_text_for_speech(text: str) -> str:
    """
    Normalizes text for natural TTS pronunciation:
    - Converts rupee amounts (₹55,000 -> fifty-five thousand rupees)
    - Converts percentages (100% -> one hundred percent)
    - Converts months and time durations
    - Strips markdown and extraneous punctuation
    """
    if not text:
        return ""

    # Replace currency amounts dynamically (e.g., ₹55000, Rs. 45,000, 40000 INR)
    def _currency_replace(match):
        num_str = re.sub(r"[^\d]", "", match.group(2))
        try:
            val = int(num_str)
            words = _int_to_words(val)
            return f"{words} rupees"
        except Exception:
            return match.group(0)

    text = re.sub(
        r"(₹|rs\.?|inr)\s*([\d,]+)",
        _currency_replace,
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"([\d,]+)\s*(rupees|inr)",
        _currency_replace,
        text,
        flags=re.IGNORECASE
    )

    # Specific common amounts without explicit symbol if formatted with comma (e.g. 55,000)
    for amount, word in [
        ("55,000", "fifty-five thousand"),
        ("45,000", "forty-five thousand"),
        ("40,000", "forty thousand"),
        ("2,000", "two thousand"),
    ]:
        text = re.sub(rf"\b{re.escape(amount)}\b", word, text)

    # Durations
    text = re.sub(r"\b8\s*months?\b", "eight months", text, flags=re.IGNORECASE)
    text = re.sub(r"\b6\s*months?\b", "six months", text, flags=re.IGNORECASE)
    text = re.sub(r"\b100\s*%", "one hundred percent", text)

    # Clean markdown asterisks, hashes, and formatting
    text = re.sub(r"[*#_~`]", "", text)
    # Normalize multiple whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_text(text: str, max_chars: int = 250) -> list:
    """
    Split text into sentence chunks. Uses max_chars=250 to avoid
    unnecessary chunk fragmentation on M1 CPU.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current) + len(sentence) <= max_chars:
            current += (" " if current else "") + sentence
        else:
            if current:
                chunks.append(current)
            current = sentence

    if current:
        chunks.append(current)

    return chunks or [text]


def combine_wav_files(input_files, output_file):
    """Combine multiple WAV files into a single WAV with small pause."""
    if not input_files:
        raise ValueError("No audio chunks were generated.")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if len(input_files) == 1:
        shutil.copyfile(str(input_files[0]), str(output_file))
        return

    with wave.open(str(input_files[0]), "rb") as first:
        params = first.getparams()
        frames = [first.readframes(first.getnframes())]

    for file in input_files[1:]:
        with wave.open(str(file), "rb") as wav:
            if (
                wav.getframerate() != params.framerate
                or wav.getnchannels() != params.nchannels
                or wav.getsampwidth() != params.sampwidth
                or wav.getcomptype() != params.comptype
            ):
                raise ValueError(
                    f"Incompatible audio format in {file}"
                )

            # Add 150 milliseconds of silence between chunks
            silence_frames = int(params.framerate * 0.15)
            silence = b"\x00" * (
                silence_frames
                * params.nchannels
                * params.sampwidth
            )

            frames.append(silence)
            frames.append(wav.readframes(wav.getnframes()))

    with wave.open(str(output_file), "wb") as output:
        output.setparams(params)
        for frame_data in frames:
            output.writeframesraw(frame_data)


def _synthesize_kokoro(text: str, output_path: Path):
    """Fallback / fast Apple Silicon Metal TTS using Kokoro-MLX."""
    global _kokoro_instance
    if _kokoro_instance is None:
        from kokoro_mlx import KokoroTTS
        _kokoro_instance = KokoroTTS.from_pretrained()

    _kokoro_instance.save(
        text,
        str(output_path),
        voice="af_heart",
        speed=1.0,
        sample_rate=24000
    )
    return str(output_path)


def text_to_speech(text: str, output_path: str = "response.wav") -> str:
    """
    Generate speech audio for input text.
    Handles chunking, session isolation, and clean temporary file deletion.
    """
    speech_text = prepare_text_for_speech(text)

    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # If Kokoro engine is explicitly selected
    if TTS_ENGINE in ("kokoro", "mlx"):
        return _synthesize_kokoro(speech_text, output_path)

    chunks = split_text(speech_text, max_chars=250)
    session_id = uuid.uuid4().hex[:8]
    chunk_files = []

    try:
        for index, chunk in enumerate(chunks):
            chunk_file = output_path.parent / f"_chunk_{session_id}_{index}.wav"

            command = [
                "f5-tts_infer-cli",
                "--model", "F5TTS_v1_Base",
                "--ref_audio", str(REFERENCE_AUDIO),
                "--ref_text", REFERENCE_TEXT,
                "--gen_text", chunk,
                "--output_file", str(chunk_file),
                "--nfe_step", "16",
                "--cfg_strength", "2.0",
                "--speed", "0.90",
                "--device", "cpu",
            ]

            try:
                subprocess.run(command, check=True, capture_output=True)
                chunk_files.append(chunk_file)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"[TTS Warning] F5-TTS inference failed: {e}. Attempting Kokoro fallback...")
                try:
                    return _synthesize_kokoro(speech_text, output_path)
                except Exception as fb_err:
                    raise RuntimeError(f"TTS synthesis failed on both F5-TTS and Kokoro: {fb_err}") from e

        combine_wav_files(chunk_files, output_path)
        return str(output_path)

    finally:
        # Guarantee cleanup of all temporary chunk files
        for cf in chunk_files:
            try:
                if cf.exists():
                    cf.unlink()
            except OSError:
                pass