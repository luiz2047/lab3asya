import json
import random


class DialogueManager:
    def __init__(self):
        with open('datasets/intents.json', 'r') as file:
            self.intents = json.load(file)

    def respond(self, text, intent):
        label_map = {intent['tag']: idx for idx, intent in enumerate(self.intents['intents'])}
        intent = get_key(intent, label_map)
        for intent_data in self.intents['intents']:
            if intent_data['tag'] == intent:
                return intent_data['responses'][random.choice(range(len(intent_data['responses'])))]
        return "Sorry, I didn't understand that."


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key

    return "key doesn't exist"
