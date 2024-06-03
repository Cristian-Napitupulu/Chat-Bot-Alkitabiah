import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pickle
import json
import os
import random
import time

# Get the current directory
current_directory = os.getcwd()

# Load intents data from JSON
with open(
    os.path.join(current_directory, "data/intents/new_intents.json"), "r"
) as file:
    data = json.load(file)

gpu_name = "/GPU:0"
cpu_name = "/CPU:0"
# # Choose device for prediction
# device_name = input("Enter device to use for prediction (CPU or GPU): ").strip().upper()
# if device_name == "GPU" and tf.config.list_physical_devices("GPU"):
#     device = gpu_name
# elif device_name == "GPU" and not tf.config.list_physical_devices("GPU"):
#     print("No GPU detected. Using CPU instead.")
#     device = cpu_name
#     time.sleep(2)
# else:
#     device = cpu_name

device = gpu_name if tf.config.list_physical_devices("GPU") else cpu_name

device_ = "GPU" if str(device) == gpu_name else "CPU"

# Load the model and tokenizers
model = load_model(os.path.join(current_directory, "neural_network_chatbot_model/intent_model.h5"))

with open(os.path.join(current_directory, "neural_network_chatbot_model/tokenizer.pickle"), "rb") as handle:
    tokenizer = pickle.load(handle)

with open(os.path.join(current_directory, "neural_network_chatbot_model/label_encoder.pickle"), "rb") as handle:
    label_encoder = pickle.load(handle)

categories = label_encoder.classes_

def predict_intent(text, device):
    with tf.device(device):
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(
            sequence, maxlen=model.input_shape[1], padding="post"
        )
        prediction = model.predict(padded_sequence)
        tag_index = np.argmax(prediction)
        tag = label_encoder.inverse_transform([tag_index])[0]
        return tag

def get_response(message):
    intent = predict_intent(message, device)
    for intent_data in data["intents"]:
        if intent_data["tag"] == intent:
            response = random.choice(intent_data["responses"])
            break
    return f"Intent: {intent}\nResponse: {response}\n\nModel: Neural Network \nDevice: {device_}"


if __name__ == "__main__":
    # Load test questions from JSON
    with open(
        os.path.join(current_directory, "data/intents/test_patterns.json"), "r"
    ) as file:
        test_questions = json.load(file)

    for question in test_questions:
        print(f"\nQuestion: {question}")
        print(get_response(question))
        print("\n-------------------------")

    print("Finished...")
