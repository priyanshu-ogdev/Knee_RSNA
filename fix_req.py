import os

path = 'requirements.txt'
with open(path, 'a', encoding='utf-8') as f:
    f.write('\n# 6. NLP Tokenizer Support\ntiktoken\nsentencepiece\n')

print("Appended tiktoken and sentencepiece to requirements.txt")
