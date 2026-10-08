import os

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Change the argparse default for --model
target = 'parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-72B-Instruct", help="LLM HuggingFace ID")'
replacement = 'parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct"), help="LLM HuggingFace ID")'

text = text.replace(target, replacement)

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Fixed argparse default for --model to respect LLM_MODEL_ID")
