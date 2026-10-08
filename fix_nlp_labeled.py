import re

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

target = '''                      "negative": int(((final_df[target] >= 0.0) & (final_df[target] < 0.5)).sum()),
                      "masked": int((final_df[target] < 0.0).sum()),'''
replacement = '''                      "negative": int(((final_df[target] >= 0.0) & (final_df[target] < 0.5)).sum()),
                      "masked": int((final_df[target] < 0.0).sum()),
                      "labeled": int((final_df[f'{target}_weight'] > 0).sum()),'''

text = text.replace(target, replacement)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed nlp_extractor.py labeled")
