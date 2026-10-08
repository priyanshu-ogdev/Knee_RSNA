import re

path = r'src\data\preprocess\runner.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('336 px', '518 px')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed runner.py")
