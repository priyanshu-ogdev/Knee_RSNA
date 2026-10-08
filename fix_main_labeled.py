import re

path = r'src\main.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

target = '''            "labeled": int(labels_df[target].notna().sum()),
            "weighted": int((labels_df[f"{target}_weight"] > 0).sum()),'''
replacement = '''            "labeled": int((labels_df[f"{target}_weight"] > 0).sum()),'''

text = text.replace(target, replacement)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed main.py labeled")
