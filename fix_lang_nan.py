import re
import os

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

target = 'def detect_language(report: str) -> str:\n    r_lower = report.lower()'
replacement = 'def detect_language(report: str) -> str:\n    if not isinstance(report, str):\n        return "English"\n    r_lower = report.lower()'

text = text.replace(target, replacement)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed detect_language NaN handling")
