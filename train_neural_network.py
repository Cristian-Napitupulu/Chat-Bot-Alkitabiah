import json
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Dense, Dropout  # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from sklearn.preprocessing import LabelEncoder
import pickle

# Get the current directory
current_directory = os.getcwd()

# Load intents data from JSON
with open(os.path.join(current_directory, "data/intents/new_intents.json"), "r") as file:
    data = json.load(file)

# Extract patterns and tags
patterns = []
tags = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        patterns.append(pattern)
        tags.append(intent['tag'])

# Tokenize the patterns
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)
padded_sequences = pad_sequences(sequences, padding='post')

# Encode the output labels
label_encoder = LabelEncoder()
encoded_tags = label_encoder.fit_transform(tags)

# Convert labels to categorical
num_classes = len(np.unique(encoded_tags))
categorical_tags = tf.keras.utils.to_categorical(encoded_tags, num_classes=num_classes)

# Create the model
def create_model(input_shape, num_classes):
    model = Sequential([
        Dense(128, input_shape=(input_shape,), activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model(padded_sequences.shape[1], num_classes)
model.summary()
model.fit(padded_sequences, categorical_tags, epochs=500, verbose=1)

output_dir = "./neural_network_chatbot_model"
# Save the model and tokenizers
model.save(os.path.join(current_directory, output_dir +"/intent_model.h5"))

with open(os.path.join(current_directory,  output_dir +'/tokenizer.pickle'), 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(os.path.join(current_directory, output_dir +'/label_encoder.pickle'), 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

print()
print("Model", output_dir, "was saved.")
print("Finished.")