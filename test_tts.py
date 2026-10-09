from kokoro_mlx import KokoroTTS

tts = KokoroTTS.from_pretrained()

tts.speak(
    "Hello! Welcome to Foximize Technologies. How can I help you today?"
)