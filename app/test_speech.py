from app.tts import text_to_speech


text = """
Hello, welcome to Foximize Technologies.
Our Data Science with Gen AI course is an eight-month program,
and the course fee is fifty-five thousand rupees.
"""


output_file = "data/test_kokoro.wav"

text_to_speech(
    text,
    output_file
)

print("Speech generated successfully.")
print(f"Audio file: {output_file}")