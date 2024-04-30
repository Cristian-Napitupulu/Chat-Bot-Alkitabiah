import numpy as np
import pandas as pd
import json

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# specify GPU
# device = torch.device("cuda")
# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# We have prepared a chitchat dataset with 5 labels
df = pd.read_excel("./dataset/patterns_and_tags.xlsx")
df = df[df['tag'] != "no-response"]
print (df.head())

# buat sedemikian rupa sehingga untuk setiap label terdapat sekitar 100 buah
print (df['tag'].value_counts())

# Converting the labels into encodings
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['label'] = le.fit_transform(df['tag'])

# check class distribution
print (df['label'].value_counts(normalize = True))

categories = np.unique(list(df['tag']))
print (categories)

# Convert categories to Python list
categories_list = categories.tolist()
with open('./dataset/label_list.json', 'w') as file:
    json.dump(categories_list, file, indent=4)
print("JSON file created successfully.")

# In this example we have used all the utterances for training purpose
train_text, train_labels = list(df['text']), list(df['label'])
# for text in train_text:
#     try:
#         print("Text: "+ text +" \t Type: ", type(text))
#     except:
#         print(text)
#         pass
# print (train_labels)

#-------------------------------------------------

# Split the dataset into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_text, train_labels, random_state=42, test_size=0.2
)

print (type(train_texts))
print (type(val_texts))

#import sys
#sys.exit()


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

# Train the model
trainer.train()

# Save the fine-tuned model
output_dir = "./fine_tuned_chatbot-bert_model"
model.save_pretrained(output_dir)


print ()
print ("Model ", output_dir, " was saved ....")
print ("Selesai ...")
