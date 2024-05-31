import speech_recognition as sr
from pydub import AudioSegment
import os


class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def recognize_speech(self, audio_file):
        wav_file = audio_file.replace('.ogg', '.wav')
        AudioSegment.from_ogg(audio_file).export(wav_file, format='wav')

        with sr.AudioFile(wav_file) as source:
            audio = self.recognizer.record(source)
        os.remove(wav_file)

        try:
            text = self.recognizer.recognize_google(audio, language="ru-RU")
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand the audio."
        except sr.RequestError as e:
            return f"Could not request results; {e}"
