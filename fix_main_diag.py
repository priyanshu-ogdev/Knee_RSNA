import re

path = r'src\main.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

target = '"positive": int((labels_df[target] > 0).sum()),'
replacement = '"positive": int((labels_df[target] >= 0.5).sum()),'

text = text.replace(target, replacement)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed src/main.py diagnostics")
