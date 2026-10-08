import os
path = r'src\main.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('default="auto", choices=["auto", "vllm", "rules"]', 'default="vllm", choices=["vllm"]')
text = text.replace('default="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"', 'default="Qwen/Qwen2.5-72B-Instruct"')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print('Replaced in main.py')

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace('if is_present:', 'if (is_present or is_absent) and original_report:')
text = text.replace('out[t], out[f"{t}_weight"] = 0.0, 0.0 # Soft Negative', 'out[t], out[f"{t}_weight"] = 0.05, 0.1 # Soft Negative')

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print('Replaced in nlp_extractor.py')
