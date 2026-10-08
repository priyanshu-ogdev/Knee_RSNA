import re

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

start = text.find('def build_prompt')
end = text.rfind('def clean_txt')
dup_chunk = text[start:end]

# What top level functions are inside this dup_chunk?
import ast
try:
    tree = ast.parse(dup_chunk)
    funcs = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    print(f"Functions found: {funcs}")
except SyntaxError:
    print("SyntaxError parsing the chunk. Let's use regex.")
    funcs = re.findall(r'^def (\w+)', dup_chunk, re.MULTILINE)
    print(f"Regex functions found: {funcs}")
