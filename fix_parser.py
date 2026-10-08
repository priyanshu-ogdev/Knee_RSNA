import re

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

target = 's_low = raw_state.lower().strip()'
replacement = "s_low = raw_state.lower().strip(' .,\"\\'\\n')"

text = text.replace(target, replacement)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed parse_json_response trailing punctuation")
