import re

path = r'src\data\preprocess\sampling.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('~4 of 11 windows,\n    i.e. ~12 of the 24 stored slices', '~8 of 22 windows,\n    i.e. ~24 of the 24 stored slices')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed sampling.py")
