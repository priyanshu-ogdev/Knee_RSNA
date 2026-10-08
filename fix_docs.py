import os

path = r'docs\architecture.md'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Fix depth 32 -> 24 and 227 GB -> 170 GB
text = text.replace('depth 32', 'depth 24')
text = text.replace('227 GB', '170 GB')

# Remove ml_model_design.md reference
import re
text = re.sub(r' Training details and experiment\nlimitations are documented in \[ml_model_design.md\]\(ml_model_design.md\).', '', text)
text = re.sub(r' Training details and experiment\r?\nlimitations are documented in \[ml_model_design.md\]\(ml_model_design.md\).', '', text)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed architecture.md")
