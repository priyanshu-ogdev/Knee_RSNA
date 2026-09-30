import re

with open("page.html", "r", encoding="utf-8") as f:
    html = f.read()

# Let's extract script tags containing 'window.Kaggle' or similar
scripts = re.findall(r'<script.*?>.*?</script>', html, re.DOTALL)
for s in scripts:
    if 'score' in s.lower() and 'rsna-knee' in s.lower():
        print("Found data in script tag:", len(s))
        with open("script_data.txt", "w", encoding="utf-8") as out:
            out.write(s)

# Also let's just use regex to find all "Score: x.xxx" and the preceding href
matches = re.findall(r'href="(/[^/]+/[^/]+)".*?Score:\s*([0-9.]+)', html, re.DOTALL)
print("Regex matches:", len(matches))
if matches:
    print(matches[:5])
