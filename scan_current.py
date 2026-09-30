import os
import json
from playwright.sync_api import sync_playwright
import time
from bs4 import BeautifulSoup
import re

base_dir = r"D:\Knee_RSNa\all_top_100_notebooks"
if not os.path.exists(base_dir):
    print("Dir not found")
    exit()

def get_notebook_score(ref):
    url = f"https://www.kaggle.com/code/{ref}"
    print(f"Scraping score for {url} ...", flush=True)
    score = None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle")
            time.sleep(3)
            html = page.content()
            soup = BeautifulSoup(html, "html.parser")
            
            score_elements = soup.find_all(string=re.compile(r'Score:\s*0\.[0-9]+'))
            for s in score_elements:
                match = re.search(r'Score:\s*(0\.[0-9]+)', s)
                if match:
                    score = float(match.group(1))
                    break
            
            if score is None:
                text = soup.get_text()
                # look for "Best Score 0.941" etc
                matches = re.findall(r'Score.*?([0]\.[89][0-9]{2,4})', text, re.IGNORECASE)
                if matches:
                    score = float(matches[0])
            browser.close()
    except Exception as e:
        print(e)
    return score

training_notebooks = []

for d in os.listdir(base_dir):
    d_path = os.path.join(base_dir, d)
    if os.path.isdir(d_path):
        nb_file = [f for f in os.listdir(d_path) if f.endswith('.ipynb')]
        if nb_file:
            path = os.path.join(d_path, nb_file[0])
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    nb = json.load(f)
                    code_content = ""
                    for cell in nb.get('cells', []):
                        if cell.get('cell_type') == 'code':
                            code_content += "".join(cell.get('source', [])) + "\n"
                            
                    is_training = any(kw in code_content for kw in ['backward()', 'optimizer.step()', 'Trainer(', 'model.fit(', 'train_one_epoch', 'fit_one_cycle'])
                    has_finetune = any(kw in code_content for kw in ['requires_grad = False', 'unfreeze', 'finetune'])
                    
                    if is_training:
                        # try to get original ref by finding it from the kernel json metadata
                        # or just by name
                        # Actually we can't get the author easily from just the directory name, we need the kernel-metadata.json
                        # Let's see if kernel-metadata.json exists
                        meta_file = os.path.join(d_path, "kernel-metadata.json")
                        ref = d
                        if os.path.exists(meta_file):
                            with open(meta_file, 'r', encoding='utf-8') as mf:
                                meta = json.load(mf)
                                ref = meta.get('id', d)
                        else:
                            # If no meta, maybe the name is unique enough, but we need author for url.
                            # The URL needs author/slug.
                            print(f"Skipping {d} due to missing metadata for URL.")
                            continue
                            
                        print(f"Found training code in {ref}. Checking score...", flush=True)
                        score = get_notebook_score(ref)
                        training_notebooks.append((ref, score))
                        print(f" -> Score: {score}")
            except Exception as e:
                pass

print("--- ALL FOUND TRAINING NOTEBOOKS ---")
for ref, score in sorted(training_notebooks, key=lambda x: x[1] if x[1] else 0, reverse=True):
    print(f"Score: {score} | Ref: {ref}")
