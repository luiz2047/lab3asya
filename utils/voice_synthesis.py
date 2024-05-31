from gtts import gTTS
import os

def text_to_speech(text, output_file):
    tts = gTTS(text)
    tts.save(output_file)
    os.system(f"mpg321 {output_file}")
