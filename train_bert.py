import numpy as np
import pandas as pd
import json
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Read dataset
df = pd.read_excel("./data/intents/patterns_and_tags.xlsx")
df = df[df['tag'] != "no-response"]
print(df.head())

# Ensure approximately 100 instances for each label
print(df['tag'].value_counts())

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['tag'])

# Check class distribution
print(df['label'].value_counts(normalize=True))

categories = np.unique(list(df['tag']))
print(categories)

# Save label list to JSON
categories_list = categories.tolist()
with open('./data/intents/label_list.json', 'w') as file:
    json.dump(categories_list, file, indent=4)
print("JSON file created successfully.")

# Prepare data for training
train_text, train_labels = list(df['text']), list(df['label'])

# Split dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_text, train_labels, random_state=42, test_size=0.2
)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize input texts
train_encodings = tokenizer(train_texts, padding=True, truncation=True)
val_encodings = tokenizer(val_texts, padding=True, truncation=True)

# Define PyTorch datasets
class PyTorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = PyTorchDataset(train_encodings, train_labels)
val_dataset = PyTorchDataset(val_encodings, val_labels)

# Fine-tune a pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(categories), ignore_mismatched_sizes=True)

training_args = TrainingArguments(
    output_dir="test_trainer",
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    num_train_epochs=50,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("Using device: " + str(device))
# Train the model
trainer.train()

# Save the fine-tuned model
output_dir = "./fine_tuned_chatbot-bert_model"
model.save_pretrained(output_dir)

print()
print("Model", output_dir, "was saved.")
print("Finished.")
