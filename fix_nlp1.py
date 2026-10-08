import os, re

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8', errors='ignore') as f:
    text = f.read()

# Fix parse_json_response value mapping bug
target_bug = '''              # Map verified findings to labels & confidence weights
              conf_str = str(v.get('confidence', '')).lower()
              if (is_present or is_absent) and original_report:
                  if 'high' in conf_str:
                      out[t], out[f"{t}_weight"] = 0.95, 1.0
                  elif 'low' in conf_str:
                      out[t], out[f"{t}_weight"] = 0.65, 0.5
                  else:
                      out[t], out[f"{t}_weight"] = 0.85, 0.85
              elif is_absent:'''

replacement_bug = '''              # Map verified findings to labels & confidence weights
              conf_str = str(v.get('confidence', '')).lower()
              if is_present:
                  if 'high' in conf_str:
                      out[t], out[f"{t}_weight"] = 0.95, 1.0
                  elif 'low' in conf_str:
                      out[t], out[f"{t}_weight"] = 0.65, 0.5
                  else:
                      out[t], out[f"{t}_weight"] = 0.85, 0.85
              elif is_absent:'''

text = text.replace(target_bug, replacement_bug)

# Fix detect_language completely
start_dl = text.find('def detect_language')
end_dl = text.find('if __name__ ==')
new_dl = '''def detect_language(report: str) -> str:
    r_lower = report.lower()
    if 'sağlam' in r_lower or 'yırtık' in r_lower or 'eklem' in r_lower:
        return 'Turkish'
    if 'ruptura' in r_lower and ('uredno' in r_lower or 'intaktno' in r_lower or 'lezija' in r_lower):
        return 'Croatian/Serbian'
    elif 'uredno' in r_lower or 'intaktno' in r_lower:
        return 'Croatian/Serbian'
    if 'повреда' in r_lower or 'разрыв' in r_lower or 'без' in r_lower:
        return 'Russian'
    if 'ρήξη' in r_lower or 'φυσιολογικός' in r_lower:
        return 'Greek'
    if 'rotura' in r_lower or 'derrame' in r_lower:
        return 'Spanish'
    if 'scheur' in r_lower or 'geen' in r_lower:
        return 'Dutch'
    if 'ruptur' in r_lower or 'erguss' in r_lower or 'kein' in r_lower:
        return 'German'
    if 'rupture' in r_lower or 'épanchement' in r_lower or 'sans' in r_lower:
        return 'French'
    return 'English'

'''
if start_dl != -1 and end_dl != -1:
    text = text[:start_dl] + new_dl + text[end_dl:]

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print('Fixed detect_language and parse_json_response')
