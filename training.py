import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import json


class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_intents(filepath):
    with open(filepath, 'r') as file:
        intents = json.load(file)
    return intents


def prepare_data(intents):
    texts = []
    labels = []
    label_map = {intent['tag']: idx for idx, intent in enumerate(intents['intents'])}
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            texts.append(pattern)
            labels.append(label_map[intent['tag']])
    return texts, labels, label_map


def train_intent_classifier(intents_file, model_path, num_epochs=100):
    intents = load_intents(intents_file)
    texts, labels, label_map = prepare_data(intents)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)
    print(train_texts, val_texts, train_labels, val_labels)
    train_dataset = IntentDataset(train_texts, train_labels, tokenizer)
    val_dataset = IntentDataset(val_texts, val_labels, tokenizer)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_map))

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=0,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    return label_map


if __name__ == "__main__":
    intents_file = 'datasets/intents.json'
    model_path = 'models/intent_model'
    train_intent_classifier(intents_file, model_path)
