import os

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('import kagglehub\n    import time', 'import time')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Removed top-level kagglehub import")
