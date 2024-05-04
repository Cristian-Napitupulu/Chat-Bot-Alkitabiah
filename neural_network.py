import os
os.environ["KERAS_BACKEND"] = "torch"

import json
import keras_nlp
import torch
import keras
from keras import layers
from sklearn.model_selection import train_test_split



# Load data from JSON file
with open('./dataset/new_intents.json', 'r') as file:
    data = json.load(file)

# Extract patterns and intents from the loaded data
patterns = [intent['patterns'] for intent in data['intents']]
intents = [intent['tag'] for intent in data['intents']]

# Flatten the patterns list
patterns_flat = [pattern for sublist in patterns for pattern in sublist]

# Tokenize patterns
tokenizer = Tokenizer()
tokenizer.fit_on_texts(patterns_flat)

# Convert text data to sequences of integers
sequences = tokenizer.texts_to_sequences(patterns_flat)

# Pad sequences to ensure uniform length
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Tokenize intents
intent_tokenizer = Tokenizer()
intent_tokenizer.fit_on_texts(intents)
intent_sequences = intent_tokenizer.texts_to_sequences(intents)

# Convert intents to categorical labels
num_classes = len(intent_tokenizer.word_index) + 1  # Plus one for padding
encoded_intents = tf.keras.utils.to_categorical(intent_sequences, num_classes=num_classes)

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(padded_sequences, encoded_intents, test_size=0.2, random_state=42)

# Define model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=32)

# Prediction
def predict_intent(model, tokenizer, text):
    text_sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(text_sequence, maxlen=max_length, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_intent = intent_tokenizer.index_word[np.argmax(prediction)]
    return predicted_intent

# Example usage
user_input = "What is your refund policy?"
predicted_intent = predict_intent(model, tokenizer, user_input)
print("Predicted Intent:", predicted_intent)
