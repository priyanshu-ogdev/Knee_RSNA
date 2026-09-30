from bs4 import BeautifulSoup
import re
import os
import subprocess

with open("page.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f, "html.parser")

downloaded = 0

score_elements = soup.find_all(string=re.compile(r'Score:\s*0\.[0-9]+'))

for score_el in score_elements:
    try:
        score_match = re.search(r'Score:\s*(0\.[0-9]+)', score_el)
        if score_match:
            score = float(score_match.group(1))
            if score > 0.88:
                parent = score_el.parent
                target_link = None
                for _ in range(15):
                    if parent is None:
                        break
                    links = parent.find_all("a", href=True)
                    for link in links:
                        href = link['href']
                        if href.startswith("/") and not href.startswith("/competitions") and href.count("/") >= 2:
                            target_link = href
                            break
                    if target_link:
                        break
                    parent = parent.parent
                
                if target_link:
                    ref = target_link[1:]
                    if ref.endswith("/code"):
                        ref = ref[:-5]
                    print(f"True score {score} for {ref}")
                    out_dir = os.path.join('true_top_notebooks', ref.split('/')[-1])
                    os.makedirs(out_dir, exist_ok=True)
                    subprocess.run(f'kaggle kernels pull "{ref}" -p "{out_dir}"', shell=True)
                    downloaded += 1
    except Exception as e:
        print("Error:", e)

print(f"Downloaded {downloaded} true top notebooks.")
