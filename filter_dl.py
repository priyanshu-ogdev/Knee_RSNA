import csv
import re
import os
import subprocess

def has_high_score(text):
    text = text.lower()
    matches = re.findall(r'(0?\.[89][0-9]+|0[89][0-9]{2,3})', text)
    for m in matches:
        if m.startswith('0') and not '.' in m:
            val = float('0.' + m[1:])
        else:
            try:
                val = float(m)
            except ValueError:
                continue
        if val >= 0.89:
            return True
    return False

with open('kernels_score.csv', 'r', encoding='utf-8-sig') as f:
    lines = (line for line in f if line.strip())
    reader = csv.DictReader(lines)
    
    downloaded = 0
    for row in reader:
        ref_key = [k for k in row.keys() if k and 'ref' in k.lower()][0]
        ref = row.get(ref_key)
        title = row.get('title', '')
        if not ref: continue
        
        if has_high_score(ref) or has_high_score(title):
            print(f"Match found: {ref}", flush=True)
            dirname = ref.split('/')[-1]
            out_dir = os.path.join('top_notebooks', dirname)
            os.makedirs(out_dir, exist_ok=True)
            subprocess.run(f'kaggle kernels pull "{ref}" -p "{out_dir}"', shell=True)
            downloaded += 1
            
    print(f"Downloaded {downloaded} notebooks.")
