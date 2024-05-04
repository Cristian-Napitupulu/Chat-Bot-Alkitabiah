import re
import numpy as np
import random
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
import json
import os
from torchinfo import summary

# Get the current directory
current_directory = os.getcwd()

# Navigate one folder back
parent_directory = os.path.dirname(current_directory)

# Load intents data from JSON
with open("./data/intents/new_intents.json", "r") as file:
    data = json.load(file)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device_ = "GPU" if str(device) == "cuda" else "CPU"

# Extract tags from intents
tags = [intent["tag"] for intent in data["intents"]]
categories = np.unique(tags)
print(categories)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

bert_asli = 0
# Choose the model
if bert_asli == 1:
    model_name = "bert-base-uncased"
else:
    model_name = "./fine_tuned_chatbot-bert_model"

# Load BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    model_name, num_labels=len(categories), ignore_mismatched_sizes=True
)

# Set the maximum sequence length
max_seq_len = 8

# Move the model to GPU if available
model = model.to(device)

# Summarize the model
summary(model)

# Load label list from JSON
with open("./data/intents/label_list.json", "r") as file:
    label_list = json.load(file)

# Convert labels into encodings
le = LabelEncoder()
encoded_labels = le.fit_transform(label_list)


def get_prediction(text):
    cleaned_text = re.sub(r"[^a-zA-Z ]+", "", text)

    # Tokenize input text
    inputs = tokenizer(
        cleaned_text, padding=True, truncation=True, return_tensors="pt"
    ).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Get predicted logits
    logits = outputs.logits

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Get predicted label (index of the maximum probability)
    predicted_label_index = torch.argmax(probs, dim=-1).item()

    # Get corresponding label name
    predicted_label = categories[predicted_label_index]
    return predicted_label


def get_response(message):
    intent = get_prediction(message)
    for intent_data in data["intents"]:
        if intent_data["tag"] == intent:
            response = random.choice(intent_data["responses"])
            break

    return (
        "Intent: "
        + intent
        + "\nResponse: "
        + response
        + "\n\nUsing device: "
        + str(device_)
    )


# Load test questions from JSON
with open("./data/intents/test_patterns.json", "r") as file:
    test_questions = json.load(file)

for question in test_questions:
    print()
    print(f"Question: {question}")
    print(get_response(question))
    print()
    print("-------------------------")

print("Finished...")
