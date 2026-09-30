from bs4 import BeautifulSoup
import subprocess
import os

with open("page.html", "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

score_elements = soup.find_all(string=lambda t: t and "Score:" in t)
downloaded = 0
for el in score_elements:
    try:
        score_text = el.strip()
        score = float(score_text.replace("Score:", "").strip())
        if score >= 0.88:
            parent = el.parent
            for _ in range(10): 
                if parent is None:
                    break
                # we want the main link to the notebook
                links = parent.find_all("a", href=True)
                target_link = None
                for link in links:
                    if link['href'].startswith("/") and not link['href'].startswith("/competitions") and link['href'].count("/") == 2:
                        target_link = link
                        break
                
                if target_link:
                    href = target_link['href']
                    ref = href[1:]
                    if ref.endswith("/code"):
                        ref = ref[:-5]
                    print(f"Found {ref} with true score {score}")
                    out_dir = os.path.join('true_top_notebooks', ref.split('/')[-1])
                    os.makedirs(out_dir, exist_ok=True)
                    subprocess.run(f'kaggle kernels pull "{ref}" -p "{out_dir}"', shell=True)
                    downloaded += 1
                    break
                parent = parent.parent
    except Exception as e:
        print(f"Error parsing score element: {e}")
print(f"Downloaded {downloaded} true top notebooks.")
