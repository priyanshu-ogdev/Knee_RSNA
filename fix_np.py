import os

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('import pandas as pd', 'import pandas as pd\nimport numpy as np')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Added numpy import")
