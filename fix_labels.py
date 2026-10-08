import re

path = r'src\data\labels.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

target = '"positive": int((out[t] > 0).sum()),'
replacement = '"positive": int((out[t] >= 0.5).sum()),\n                    "labeled": int((out[f\'{t}_weight\'] > 0).sum()),'

text = text.replace(target, replacement)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed labels.py")
