import re
import subprocess
import wave
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

REFERENCE_AUDIO = BASE_DIR / "data" / "voice_reference.wav"

REFERENCE_TEXT = """
Hello, welcome to Foximize Technologies. My name is Sumayya, and I’m your virtual assistant. I’m here to help you with information about our courses, training programs, admission process, fees, internships, and other services.
""".strip()


def prepare_text_for_speech(text):
    text = re.sub(
        r"(₹|rs\.?|inr)?\s*55,?000\b",
        "fifty-five thousand rupees",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"(₹|rs\.?|inr)?\s*45,?000\b",
        "forty-five thousand rupees",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"(₹|rs\.?|inr)?\s*40,?000\b",
        "forty thousand rupees",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\b8\s*months?\b",
        "eight months",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\b6\s*months?\b",
        "six months",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"(₹|rs\.?|inr)?\s*2,?000\b",
        "two thousand rupees",
        text,
        flags=re.IGNORECASE
    )

    return text.strip()


def split_text(text, max_chars=180):
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

    return chunks




def combine_wav_files(input_files, output_file):
    if not input_files:
        raise ValueError("No audio chunks were generated.")

    output_file = Path(output_file)

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

            # Add 150 milliseconds of silence between chunks.
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

def text_to_speech(text, output_path="response.wav"):

    speech_text = prepare_text_for_speech(text)

    print("Original response:", text)
    print("Speech text:", speech_text)

    chunks = split_text(speech_text)

    print(f"Generating {len(chunks)} speech chunks...")

    output_path = Path(output_path)

    if not output_path.is_absolute():
        output_path = BASE_DIR / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_files = []

    for index, chunk in enumerate(chunks):

        print(f"Generating chunk {index + 1}/{len(chunks)}:")
        print(chunk)

        chunk_file = output_path.parent / f"_chunk_{index}.wav"

        command = [
            "f5-tts_infer-cli",

            "--model", "F5TTS_v1_Base",

            "--ref_audio",
            str(REFERENCE_AUDIO),

            "--ref_text",
            REFERENCE_TEXT,

            "--gen_text",
            chunk,

            "--output_file",
            str(chunk_file),

            "--nfe_step",
            "16",

            "--cfg_strength",
            "2.0",

            "--speed",
            "0.90",

            "--device",
            "cpu",
        ]

        subprocess.run(command, check=True)

        chunk_files.append(chunk_file)

    combine_wav_files(
        chunk_files,
        output_path
    )

    # Remove temporary chunks
    #try:
            #file.unlink()
        #except Exception:
         #   pass

    print("Final audio created:", output_path)

    return str(output_path)