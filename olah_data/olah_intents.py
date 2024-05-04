import json
import pandas as pd
import os
import random

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def filter_intents_by_count(intents, lower_cut_off, upper_cut_off):
    return [intent for intent in intents if lower_cut_off <= len(intent['patterns']) <= upper_cut_off]

# Get the directory of the script
script_dir = os.path.dirname(os.path.realpath(__file__))
print("Script directory:", script_dir)

# Load the JSON data
data = load_json(os.path.join(script_dir, 'intents_v2.json'))

# Create a dictionary to store the count of patterns for each tag
patterns_count = {intent['tag']: len(intent['patterns']) for intent in data['intents']}

print(patterns_count)

# Remove tags with less than cutoff number of patterns
lower_cut_off = 5
upper_cut_off = 1000
data['intents'] = filter_intents_by_count(data['intents'], lower_cut_off, upper_cut_off)

# Print the updated JSON data
print(json.dumps(data, indent=4))

# Save the updated JSON data to a file
save_json(data, os.path.join(script_dir, 'new_intents.json'))

print("Number of tags left:", len(data['intents']))

# Load the JSON data
data = load_json(os.path.join(script_dir, 'new_intents.json'))

# Extract patterns and tags
patterns = []
tags = []
for intent in data['intents']:
    patterns.extend(intent['patterns'])
    tags.extend([intent['tag']] * len(intent['patterns']))

test_pattern = random.sample(patterns, 5)

print(test_pattern)

save_json(test_pattern, os.path.join(script_dir, 'test_pattern.json'))

# Create a DataFrame
df = pd.DataFrame({'text': patterns, 'tag': tags})

# Save DataFrame to Excel
df.to_excel(os.path.join(script_dir, 'patterns_and_tags.xlsx'), index=False)
