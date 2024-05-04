import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

df = pd.read_excel('olah_intents.xlsx')

absolute_folder_path = os.path.abspath(os.path.dirname(__file__))
relative_folder_path = "gambar/"
folder_path = os.path.join(absolute_folder_path, relative_folder_path)

# Create double plot where x axis is tags and y axis is number of unique patterns and number of unique responses
tags = df['tags'].unique()
number_of_unique_patterns = []
number_of_unique_responses = []

for tag in tags:
    number_of_unique_patterns.append(len(df[df['tags'] == tag]['patterns'].unique()))
    number_of_unique_responses.append(len(df[df['tags'] == tag]['responses'].unique()))


plot_name = "Tags vs Number of Unique Patterns and Responses"
# Create bar plot with x axis is tags and y axis is number of unique patterns and number of unique responses
fig, ax = plt.subplots(figsize=(14, 6.5))  # Adjust figsize as needed
plt.title('Tags vs Number of Unique Patterns and Responses')
bar_width = 0.35
index = np.arange(len(tags))
bar1 = ax.bar(index, number_of_unique_patterns, bar_width, label='Number of Unique Patterns')
bar2 = ax.bar([i + bar_width for i in index], number_of_unique_responses, bar_width, label='Number of Unique Responses')  # Adjust the multiplier as needed
ax.set_xticks(index+bar_width/2, labels=tags)
plt.xticks(rotation=90)
plt.xlabel("${Tags}$")
plt.ylabel("${Number\;of\;Unique\;Patterns\;and\;Responses}$")
# Increase space between x axis labels
plt.subplots_adjust(top=0.9, bottom=0.3, left=0.06, right=0.965)  # Adjust bottom padding as needed
ax.legend()
plt.grid(axis='y')

plt.savefig(folder_path + plot_name + ".svg", format="svg", transparent=True)
plt.savefig(folder_path + plot_name + ".png")

plot_name = "Tags vs Number of Unique Patterns"
plt.figure(plot_name, figsize=(14, 6.5))
plt.xlabel("${Tags}$")
plt.ylabel("${Number\;of\;Unique\;Patterns}$")
plt.title(plot_name, fontsize=16)
plt.bar(tags, number_of_unique_patterns)
plt.xticks(rotation=90)
# Increase space between x axis labels
plt.subplots_adjust(top=0.9, bottom=0.3, left=0.06, right=0.965)  # Adjust bottom padding as needed
plt.grid(axis='y')
plt.savefig(folder_path + plot_name + ".svg", format="svg", transparent=True)
plt.savefig(folder_path + plot_name + ".png")

plot_name = "Tags vs Number of Unique Responses"
plt.figure(plot_name, figsize=(14, 6.5))
plt.xlabel("${Tags}$")
plt.ylabel("${Number\;of\;Unique\;Responses}$")
plt.title(plot_name, fontsize=16)
plt.bar(tags, number_of_unique_responses)
plt.xticks(rotation=90)
# Increase space between x axis labels
plt.subplots_adjust(top=0.9, bottom=0.3, left=0.06, right=0.965)  # Adjust bottom padding as needed
plt.grid(axis='y')
plt.savefig(folder_path + plot_name + ".svg", format="svg", transparent=True)
plt.savefig(folder_path + plot_name + ".png")

plt.show()