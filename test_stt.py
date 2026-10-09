from app.stt import transcribe_audio


audio_file = "data/test.wav"

text = transcribe_audio(audio_file)

print("\nTranscribed text:")
print(text)