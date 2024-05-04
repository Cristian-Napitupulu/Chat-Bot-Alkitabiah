import json
import pandas as pd

with open ('intents.json') as file:
    data = json.load(file)


tags = []
patterns = []
responses = []
for tag in data['intents']:
    for pattern in tag['patterns']:
        for response in tag['responses']:
            # print (pattern, response)

            tags.append(tag['tag'])
            patterns.append(pattern)
            responses.append(response)

df = pd.DataFrame({'tags': tags, 'patterns': patterns, 'responses': responses})

df.to_excel('olah_intents.xlsx', index=False)