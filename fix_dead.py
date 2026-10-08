import re

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Remove CLINICAL_RULES
start = text.find('CLINICAL_RULES = {')
end = text.find('}\n', start) + 2
if start != -1:
    text = text[:start] + text[end:]
    print("Removed CLINICAL_RULES")

# Remove _legacy_run_offline_extraction
start2 = text.find('def _legacy_run_offline_extraction(')
end2 = text.find('def build_prompt', start2)
if start2 != -1 and end2 != -1:
    text = text[:start2] + text[end2:]
    print("Removed _legacy_run_offline_extraction")

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
