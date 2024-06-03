import json
import pandas as pd
import os
import random
import openpyxl
openpyxl

intents_folder = "/intents"
intents_filename = "intents.json"

def load_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def save_json(data, file_path, save_name):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    save_path = os.path.join(file_path, save_name)
    with open(save_path, 'w') as file:
        json.dump(data, file, indent=4)

def filter_intents_by_count(intents, lower_cut_off, upper_cut_off):
    return [intent for intent in intents if lower_cut_off <= len(intent['patterns']) <= upper_cut_off]

# Get the directory of the script
script_dir = os.path.dirname(os.path.realpath(__file__))
print("Script directory:", script_dir)
print("Data directory:", script_dir + intents_folder)

# Load the JSON data
original_data = load_json(os.path.join(script_dir, intents_filename))

# Create a dictionary to store the count of patterns for each tag
patterns_count = {intent['tag']: len(intent['patterns']) for intent in original_data['intents']}

# print(patterns_count)

# Remove tags with less than cutoff number of patterns
lower_cut_off = 5
upper_cut_off = 1000
new_data = {}
new_data['intents'] = filter_intents_by_count(original_data['intents'], lower_cut_off, upper_cut_off)

# Print the updated JSON data
# print(new_data)
json.dumps(new_data, indent=4)

# Save the updated JSON data to a file
save_json(new_data, script_dir + intents_folder, 'new_intents.json')
print("Number of tags left:", len(new_data['intents']))

# Extract patterns and tags
patterns = []
tags = []
for tag in new_data['intents']:
    patterns.extend(tag['patterns'])
    tags.extend([tag['tag']] * len(tag['patterns']))

test_pattern = random.sample(patterns, 5)

print(test_pattern)

save_json(test_pattern, script_dir + intents_folder, 'test_patterns.json')

# Create a DataFrame
df = pd.DataFrame({'text': patterns, 'tag': tags})

# Save DataFrame to Excel
df.to_excel(os.path.join(script_dir + intents_folder, 'patterns_and_tags.xlsx'), index=False)

tags = []
patterns = []
responses = []
for tag in new_data['intents']:
    for pattern in tag['patterns']:
        for response in tag['responses']:
            # print (pattern, response)

            tags.append(tag['tag'])
            patterns.append(pattern)
            responses.append(response)

df = pd.DataFrame({'tags': tags, 'patterns': patterns, 'responses': responses})

df.to_excel(os.path.join(script_dir + intents_folder, 'olah_intents.xlsx'), index=False)