import os

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

target4 = 'def run_offline_extraction(data_root: str, out_csv: str, model_id: str = "Qwen/Qwen2.5-72B-Instruct"):'
replacement4 = 'def run_offline_extraction(data_root: str, out_csv: str, model_id: str = None):'
text = text.replace(target4, replacement4)

target5 = 'def _legacy_run_offline_extraction(data_root: str, out_csv: str, model_id: str = "Qwen/Qwen2.5-72B-Instruct"):'
replacement5 = 'def _legacy_run_offline_extraction(data_root: str, out_csv: str, model_id: str = None):'
text = text.replace(target5, replacement5)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed legacy wrappers")
