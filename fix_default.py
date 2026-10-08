import os

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Fix auto_complete_extraction default
target2 = 'def auto_complete_extraction(\n    data_root: str,\n    out_csv: str,\n    model_id: str = "Qwen/Qwen2.5-72B-Instruct",'
replacement2 = 'def auto_complete_extraction(\n    data_root: str,\n    out_csv: str,\n    model_id: str = None,'
text = text.replace(target2, replacement2)

# Fix internal requested_model resolution
target3 = 'requested_model = (\n        model_id\n        or os.environ.get("LLM_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")\n    )'
replacement3 = 'requested_model = model_id if model_id else os.environ.get("LLM_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")'
text = text.replace(target3, replacement3)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed auto_complete_extraction default")
