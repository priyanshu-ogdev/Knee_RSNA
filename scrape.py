from playwright.sync_api import sync_playwright
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code?sort=scoreDescending", wait_until="networkidle")
    time.sleep(5)
    html = page.content()
    with open("page.html", "w", encoding="utf-8") as f:
        f.write(html)
    browser.close()
