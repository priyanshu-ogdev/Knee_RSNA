import re

path = r'src\data\preprocess\nlp_extractor.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

# Let's find the FIRST build_prompt and everything up to the LAST clean_txt
start = text.find('def build_prompt')
end = text.rfind('def clean_txt')

new_chunk = '''def build_prompt(report: str) -> str:
    return f"""You are an expert subspecialty musculoskeletal (MSK) radiologist. 
Accurately extract the presence of the following 12 knee conditions from this MRI radiology report.

OUTPUT SCHEMA (MUST OUTPUT EXACTLY THIS JSON FORMAT WITH ALL 12 KEYS):
{{
  "ACL": {{"reasoning": "...", "exact_quote": "...", "state": "...", "confidence": "..."}},
  "MCL": {{"reasoning": "...", "exact_quote": "...", "state": "...", "confidence": "..."}},
  "Medial Meniscus": {{"reasoning": "...", "exact_quote": "...", "state": "...", "confidence": "..."}},
  "Lateral Meniscus": {{"reasoning": "...", "exact_quote": "...", "state": "...", "confidence": "..."}},
  "Medial OA": {{"reasoning": "...", "exact_quote": "...", "state": "...", "confidence": "..."}},
  "Lateral OA": {{"reasoning": "...", "exact_quote": "...", "state": "...", "confidence": "..."}},
  "PF OA": {{"reasoning": "...", "exact_quote": "...", "state": "...", "confidence": "..."}},
  "Effusion": {{"reasoning": "...", "exact_quote": "...", "state": "...", "confidence": "..."}},
  "Synovitis": {{"reasoning": "...", "exact_quote": "...", "state": "...", "confidence": "..."}},
  "Baker's": {{"reasoning": "...", "exact_quote": "...", "state": "...", "confidence": "..."}},
  "Contusion": {{"reasoning": "...", "exact_quote": "...", "state": "...", "confidence": "..."}},
  "Fracture": {{"reasoning": "...", "exact_quote": "...", "state": "...", "confidence": "..."}}
}}

TARGETS & CLINICAL DEFINITIONS:
1. ACL: Anterior Cruciate Ligament tear.
2. MCL: Medial Collateral Ligament tear or sprain.
3. Medial Meniscus: Medial meniscus tear. Post-operative states (meniscectomy, repairs) = absent. Grade 1/2 signal without articular extension = absent.
4. Lateral Meniscus: Lateral meniscus tear. Post-operative states (meniscectomy) = absent. Grade 1/2 signal = absent.
5. Medial OA: Medial tibiofemoral compartment osteoarthritis.
6. Lateral OA: Lateral tibiofemoral compartment osteoarthritis.
7. PF OA: Patellofemoral compartment osteoarthritis.
8. Effusion: Joint effusion.
9. Synovitis: Synovial thickening.
10. Baker's: Baker's cyst, popliteal cyst.
11. Contusion: Bone bruise.
12. Fracture: Cortical bone fracture.

GENERAL RULES:
1. Output MUST be valid JSON containing ALL 12 KEYS.
2. "reasoning": Think step-by-step. Keep it under 25 words.
3. "exact_quote": Copy verbatim. If absent, you MUST provide the quote proving it is absent. Absent is NEVER allowed when the structure isn't addressed; use "not_stated" instead.
4. "state": EXACTLY ONE of ["present", "absent", "not_stated"]. Mask prior-study comparisons as "not_stated".
5. "confidence": EXACTLY ONE of ["high", "medium", "low"].
6. IF a general "meniscal tear" or "menisci" finding is mentioned without specifying Medial or Lateral, apply the finding to BOTH Medial and Lateral Meniscus. For example, "Menisci are unremarkable" means BOTH are "absent".

MULTI-LINGUAL HINTS:
  - Turkish: yirtik=present, saglam=absent.
  - Croatian/Serbian: ruptura=present, uredno=absent.
  - Russian/Bulgarian: повреда=present, без=absent.
  - Greek: ρήξη=present, φυσιολογικός=absent.
  - German: Kreuzband=ACL, Erguss=Effusion, keine Ruptur/intakt=absent.
  - Spanish: LCA=ACL, derrame=effusion, sin rotura=absent.
  - Dutch: VKB/voorste kruisband=ACL, hydrops=effusion, geen scheur=absent.
  - French: LCA=ACL, épanchement=effusion, sans fissure=absent.
  
  EXAMPLES (Always output all 12 keys):
Report (English): "Anterior cruciate ligament is completely torn. Menisci are unremarkable. Minimal physiological joint fluid."
Output:
{{
  "ACL": {{"reasoning": "Explicit complete tear stated.", "exact_quote": "Anterior cruciate ligament is completely torn.", "state": "present", "confidence": "high"}},
  "MCL": {{"reasoning": "Not addressed.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Medial Meniscus": {{"reasoning": "Menisci stated as unremarkable.", "exact_quote": "Menisci are unremarkable.", "state": "absent", "confidence": "high"}},
  "Lateral Meniscus": {{"reasoning": "Menisci stated as unremarkable.", "exact_quote": "Menisci are unremarkable.", "state": "absent", "confidence": "high"}},
  "Medial OA": {{"reasoning": "Not addressed.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Lateral OA": {{"reasoning": "Not addressed.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "PF OA": {{"reasoning": "Not addressed.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Effusion": {{"reasoning": "Fluid is minimal and physiological.", "exact_quote": "Minimal physiological joint fluid.", "state": "absent", "confidence": "high"}},
  "Synovitis": {{"reasoning": "Not addressed.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Baker's": {{"reasoning": "Not addressed.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Contusion": {{"reasoning": "Not addressed.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Fracture": {{"reasoning": "Not addressed.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}}
}}

REPORT:
{report}
"""

'''

text = text[:start] + new_chunk + text[end:]

with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Removed duplicated build_prompt and old functions, inserted clean prompt.")
