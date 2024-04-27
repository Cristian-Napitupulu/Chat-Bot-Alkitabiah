# https://medium.com/geekculture/simple-chatbot-using-bert-and-pytorch-part-2-ef48506a4105
# pip install torchinfo

import re
import numpy as np
import random
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np 


# used a dictionary to represent an intents JSON file
data = {"intents": [
{"tag": "greeting",
 "responses": ["Howdy Partner!", "Hello", "How are you doing?",   "Greetings!", "How do you do?"]},
{"tag": "age",
 "responses": ["I am 25 years old", "I was born in 1998", "My birthday is July 3rd and I was born in 1998", "03/07/1998"]},
{"tag": "date",
 "responses": ["I am available all week", "I don't have any plans",  "I am not busy"]},
{"tag": "name",
 "responses": ["My name is James", "I'm James", "James"]},
{"tag": "goodbye",
 "responses": ["It was nice speaking to you", "See you later", "Speak soon!"]}
]}

# Extract tags from the intents
tags = [intent["tag"] for intent in data["intents"]]
categories = np.unique(tags)
print (categories)

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


bert_asli=1

if (bert_asli==1):
	nama_model = 'bert-base-uncased'
else:
	nama_model = "./fine_tuned_chatbot-bert_model"

model = BertForSequenceClassification.from_pretrained(nama_model, num_labels=len(categories))

# Based on the histogram we are selecting the max len as 8
max_seq_len = 8


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# push the model to GPU
model = model.to(device)

from torchinfo import summary
summary(model)


#-----------------------------------


label_list = ['date', 'name', 'date', 'date', 'goodbye']

# Converting the labels into encodings
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
embuh = le.fit_transform(label_list)


def get_predictionX(str):
 str = re.sub(r'[^a-zA-Z ]+', '', str)
 test_text = [str]
 model.eval()
 
 tokens_test_data = tokenizer(test_text, max_length = max_seq_len, pad_to_max_length=True, truncation=True,
 return_token_type_ids=False
 )

 test_seq = torch.tensor(tokens_test_data['input_ids'])
 test_mask = torch.tensor(tokens_test_data['attention_mask'])
 
 preds = None
 with torch.no_grad():
   preds = model(test_seq.to(device), test_mask.to(device))

 preds = preds.detach().cpu().numpy()
 preds = np.argmax(preds, axis = 1)

 print("Intent Identified: ", le.inverse_transform(preds)[0])
 return le.inverse_transform(preds)[0]


def get_prediction(str):
	example_text = re.sub(r'[^a-zA-Z ]+', '', str)

	# Tokenize input text
	inputs = tokenizer(example_text, padding=True, truncation=True, return_tensors='pt')

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
  for i in data['intents']: 
    if i["tag"] == intent:
      result = random.choice(i["responses"])
      break
  # print(f"Response : {result}")
  return "Intent: "+ intent + '\n' + "Response: " + result

pertanyaan_list = ["who are you", "how old are you", "do you know what is today", "goodbye"]
for pertanyaan in pertanyaan_list:
	print()
	print(f"Question : {pertanyaan}")
	print (get_response(pertanyaan))


print ("Selesai...")
