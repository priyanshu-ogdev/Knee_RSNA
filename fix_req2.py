import os

path = 'requirements.txt'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('transformers\n', 'transformers>=4.45.0\n')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Updated transformers to >=4.45.0")
