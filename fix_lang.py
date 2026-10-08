import re
import os

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Let's extract the detect_language function and replace it entirely
start = text.find('def detect_language(report: str) -> str:')
end = text.find('if __name__ == ', start)

new_func = '''def detect_language(report: str) -> str:
    r_lower = report.lower()
    
    # Use word boundaries to prevent substring matches (e.g. "ruptur" inside "rupture")
    def has_words(words):
        return any(re.search(rf'\\b{w}\\b', r_lower) for w in words)
        
    if has_words(["sağlam", "yırtık", "eklem"]):
        return "Turkish"
        
    # Croatian/Serbian
    if has_words(["uredno", "intaktno", "lezija", "koljena", "zglob"]) or (has_words(["ruptura"]) and not has_words(["ligament", "tear"])):
        return "Croatian/Serbian"
        
    # Russian
    if has_words(["повреда", "без", "разрыв"]):
        return "Russian"
        
    # Greek
    if has_words(["ρήξη", "φυσιολογικός"]):
        return "Greek"
        
    # Spanish
    if has_words(["rotura", "derrame", "sin", "rodilla"]):
        return "Spanish"
        
    # Dutch
    if has_words(["scheur", "geen", "voorste", "kruisband"]):
        return "Dutch"
        
    # German
    if has_words(["ruptur", "erguss", "kein", "kreuzband"]):
        return "German"
        
    # French
    if has_words(["rupture", "épanchement", "sans", "fissure"]):
        # Disambiguate from English 'rupture'
        if has_words(["sans", "épanchement", "genou", "fissure"]):
            return "French"
            
    return "English"

'''

if start != -1 and end != -1:
    text = text[:start] + new_func + text[end:]
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Fixed detect_language")
else:
    print("Could not find detect_language bounds")
