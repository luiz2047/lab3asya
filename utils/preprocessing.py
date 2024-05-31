import json

def load_intents(filepath):
    with open(filepath, 'r') as file:
        intents = json.load(file)
    return intents

def preprocess_text(text):
    # Add any text preprocessing steps here
    return text.lower()
