from html.parser import HTMLParser
import re

class KaggleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_content = []
        
    def handle_data(self, data):
        data = data.strip()
        if data:
            self.text_content.append(data)

with open("page.html", "r", encoding="utf-8") as f:
    content = f.read()

parser = KaggleParser()
parser.feed(content)

with open("parsed.txt", "w", encoding="utf-8") as f:
    for text in parser.text_content:
        f.write(text + "\n")
