from kokoro_mlx import KokoroTTS

# Load once when the application starts
tts = KokoroTTS.from_pretrained()


def text_to_speech(
    text,
    output_path="response.wav"
):
    tts.save(
        text,
        output_path,
        voice="af_heart",
        speed=1.0,
        sample_rate=24000
    )

    return output_path