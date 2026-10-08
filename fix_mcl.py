import re
import os

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

target = '2. MCL: Medial Collateral Ligament tear or sprain.'
replacement = '2. MCL: Medial Collateral Ligament tear or sprain. Peri-ligamentous edema = present.'
text = text.replace(target, replacement)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed MCL rule")
