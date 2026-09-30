from bs4 import BeautifulSoup
import re
import os
import subprocess

with open("page.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

downloaded = 0
score_elements = soup.find_all(string=re.compile(r'Score:\s*0\.[0-9]+'))
notebooks_to_pull = {}

for score_el in score_elements:
    try:
        score_match = re.search(r'Score:\s*(0\.[0-9]+)', score_el)
        if score_match:
            score = float(score_match.group(1))
            if score >= 0.88:
                parent = score_el.parent
                ref = None
                for _ in range(15):
                    if parent is None:
                        break
                    links = parent.find_all("a", href=True)
                    for link in links:
                        href = link['href']
                        if "/code" in href or (href.startswith("/") and href.count("/") >= 2 and not href.startswith("/competitions")):
                            parts = [p for p in href.split('/') if p and p != 'code' and p != 'comments' and p != 'competitions']
                            if len(parts) >= 2:
                                ref = f"{parts[0]}/{parts[1]}"
                                notebooks_to_pull[ref] = score
                                break
                    if ref:
                        break
                    parent = parent.parent
    except Exception as e:
        print("Error:", e)

for ref, score in notebooks_to_pull.items():
    print(f"True score {score} for {ref}")
    out_dir = os.path.join('true_top_notebooks', ref.split('/')[-1])
    os.makedirs(out_dir, exist_ok=True)
    subprocess.run(f'kaggle kernels pull "{ref}" -p "{out_dir}"', shell=True)
    downloaded += 1

print(f"Downloaded {downloaded} true top notebooks.")
