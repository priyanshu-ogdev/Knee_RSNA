import re

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

start = text.find('def _legacy_run_offline_extraction')
if start != -1:
    # Find the next top-level def or the end of the file
    end = text.find('\ndef ', start + 5)
    if end == -1:
        end = len(text)
    
    text = text[:start] + text[end:]
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Removed _legacy_run_offline_extraction")
else:
    print("Not found")
