import os
import json
import re
import hashlib
import time
import gc
import ast
import argparse
import psutil
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_fscore_support

import sys
# Module-level project root resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import src.core.config as config
from src.core.config import resolve_data_root

# SOTA Fix: Robustly load .env from project root so HF_TOKEN / HUGGING_FACE_HUB_TOKEN is always accessible.
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(PROJECT_ROOT, ".env")
    if os.path.exists(_env_path):
        load_dotenv(_env_path)
    load_dotenv()
    if os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        os.environ["HUGGING_FACE_HUB_TOKEN"] = os.environ["HF_TOKEN"]
    elif os.environ.get("HUGGING_FACE_HUB_TOKEN") and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = os.environ["HUGGING_FACE_HUB_TOKEN"]
except ImportError:
    pass

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM, SamplingParams = None, None

# Compatibility hook for legacy testing / rule mock
extract_by_rules = None

# The exact targets expected by the training pipeline
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", 
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]

EXTRACTOR_VERSION = "clinical-report-labels-v4"

# High-throughput batching & context defaults
DEFAULT_CHUNK_SIZE = int(os.environ.get("NLP_CHUNK_SIZE", "150"))
DEFAULT_MAX_MODEL_LEN = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
DEFAULT_MAX_TOKENS = int(os.environ.get("NLP_MAX_TOKENS", "2048"))


def append_to_jsonl(uid: str, raw_output: str, out_csv: str) -> None:
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "raw_llm_outputs.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"StudyInstanceUID": uid, "raw_output": raw_output}) + "\n")


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: str, payload: dict) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_csv(frame: pd.DataFrame, path: str) -> None:
    temporary = f"{path}.tmp"
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _report_sha256(report: str) -> str:
    return hashlib.sha256(str(report).encode("utf-8")).hexdigest()


def _empty_extraction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "StudyInstanceUID",
            *TARGETS,
            *(f"{target}_weight" for target in TARGETS),
            "report_sha256",
            "extractor_version",
            "engine",
            "model_id",
            "prompt_sha256",
        ]
    )


def _validated_extraction_rows(
    frame: pd.DataFrame,
    expected_uids: set[str],
    expected_report_hashes: dict[str, str] | None = None,
    expected_provenance: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Keep only complete, in-range extraction rows for the requested studies.
    
    Accepts probabilities in [0.0, 1.0] as well as -1.0 (the calibrated soft-negative marker
    for unstated non-core findings consumed by labels.py).
    """
    required = {
        "StudyInstanceUID",
        *TARGETS,
        *(f"{target}_weight" for target in TARGETS),
    }
    if expected_report_hashes is not None:
        required.add("report_sha256")
    if expected_provenance:
        required.update(expected_provenance)
    if not required.issubset(frame.columns):
        return frame.iloc[0:0].copy()

    clean = frame.copy()
    clean["StudyInstanceUID"] = clean["StudyInstanceUID"].astype(str).str.strip()
    clean = clean[clean["StudyInstanceUID"].isin(expected_uids)].copy()
    valid = clean["StudyInstanceUID"].ne("")
    for target in TARGETS:
        values = pd.to_numeric(clean[target], errors="coerce")
        # SOTA: Accept [0.0, 1.0] probabilities or -1.0 soft-negative unstated marker
        valid &= values.notna() & (values.between(0.0, 1.0) | values.eq(-1.0))
        clean[target] = values
        weight_column = f"{target}_weight"
        weights = pd.to_numeric(clean[weight_column], errors="coerce")
        valid &= weights.notna() & weights.ge(0.0) & weights.le(1.0)
        clean[weight_column] = weights
    if expected_report_hashes is not None:
        expected_hash = clean["StudyInstanceUID"].map(expected_report_hashes)
        valid &= clean["report_sha256"].astype(str).eq(expected_hash.astype(str))
    for column, expected in (expected_provenance or {}).items():
        valid &= clean[column].astype(str).eq(str(expected))
    return clean.loc[valid].drop_duplicates("StudyInstanceUID", keep="last")


def build_prompt(report: str) -> str:
    # Truncate extremely long anomalies to prevent vLLM max context crashes (8192 token limit)
    # A standard knee MRI report is < 3000 chars. 10000+ chars indicates corrupted metadata.
    if len(report) > 10000:
        report = report[:10000] + "\n...[TRUNCATED TO PREVENT VLLM CONTEXT CRASH]"
        
    return f"""You are an expert subspecialty musculoskeletal (MSK) radiologist extracting 12 knee conditions from an MRI report.
Your goal is MAXIMUM PRECISION: only mark a finding as "present" when there is unambiguous, explicit, positive evidence in the report text.
When in doubt, choose "not_stated" over "present". False positives are worse than false negatives in this task.

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
  "Fracture": {{"reasoning": "...", "exact_quote": "...", "state": "...", "confidence": "..."}},
}}

TARGETS & CLINICAL MSK DEFINITIONS:
1. ACL: Anterior Cruciate Ligament tear (partial, high-grade, complete, rupture). Rules:
   - Intact/normal/unremarkable ACL = absent.
   - Intact ACL reconstruction graft = absent. Disrupted/re-torn graft = present.
   - Mucoid degeneration WITHOUT disruption = absent.
   - ACL described as "normal", "intact", "in normal course and signal" = absent (NEVER not_stated when explicitly described as normal).
   - CRITICAL: If the report evaluates the ACL and finds it normal, mark absent. Do NOT mark not_stated just because the report is in another language.

2. MCL: Medial Collateral Ligament tear or sprain (Grade 1/2/3), ONLY if the tear/edema is EXPLICITLY attributed to the MEDIAL COLLATERAL LIGAMENT (MCL) or "inner/medial collateral ligament".
   - Peri-ligamentous edema = present ONLY if text explicitly says "MCL", "medial collateral ligament", "inneres Seitenband", "ligamentum collaterale mediale", "unutarasnjivalo" or similar.
   - Generic "peri-ligamentous edema" without specifying MCL = NOT MCL (mark not_stated).
   - Peri-ACL, peri-PCL, peri-fibular, or peri-lateral collateral edema = NOT MCL.
   - MPFL tears = NOT MCL (patellofemoral, different structure).
   - "Collateral ligaments intact", "ligaments unremarkable", "lateral i medial kolateralni ligamenti uredni" = absent.

3. Medial Meniscus: Medial meniscus tear (radial, horizontal, root, flap, bucket-handle, complex, oblique). Rules:
   - Grade 1/2 intrameniscal signal without articular surface extension = absent.
   - Prior partial meniscectomy with stable remnant = absent.
   - Medial meniscus extrusion ALONE without tear = absent for tear (but note the extrusion).
   - "Medial compartment OA" or "medial chondral wear" does NOT imply a meniscal tear.
   - SHARED RULE: "Menisci are intact/unremarkable/normal" = BOTH menisci absent.
   - SHARED RULE: "Bilateral meniscal tears" or "both menisci torn" = BOTH menisci present.

4. Lateral Meniscus: Lateral meniscus tear. Rules:
   - Prior meniscectomy with stable remnant = absent.
   - Grade 1/2 signal = absent. Discoid meniscus without tear = absent.
   - Lateral compartment OA or chondral wear does NOT imply a meniscal tear.
   - Apply same SHARED RULE as above.

5. Medial OA: SPECIFICALLY medial tibiofemoral compartment osteoarthritis — joint space narrowing, subchondral sclerosis, osteophytes, or chondral wear of medial femoral condyle or medial tibial plateau. Rules:
   - "OA" or "arthrosis/artrosis/artrotske promjene/Arthrose" WITHOUT specifying "medial" compartment = NOT Medial OA (mark not_stated unless context makes it clear).
   - Patellofemoral OA does NOT count as Medial OA.
   - Lateral compartment OA does NOT count as Medial OA.
   - Must have explicit "medial" compartment attribution OR describe findings on medial femoral condyle/tibial plateau.
   - CAUTION: "Gonartrosis" (Spanish, general knee OA) alone = not_stated for Medial OA unless medial compartment is specifically mentioned.
   - DO NOT assign the same OA report to ALL THREE OA targets unless each compartment is explicitly mentioned.

6. Lateral OA: SPECIFICALLY lateral tibiofemoral compartment osteoarthritis, joint space narrowing, or chondral wear of lateral femoral condyle or lateral tibial plateau. Rules:
   - Same compartment-specificity rules as Medial OA — must explicitly state lateral compartment.
   - "OA" without qualifier = not_stated.
   - "Lateral compartment chondromalacia" = present.

7. PF OA: SPECIFICALLY patellofemoral compartment osteoarthritis — patellar facet arthrosis, chondromalacia patellae, trochlear cartilage loss, retropatellar chondral injury, patellar chondrosis. Rules:
   - "Chondropathy/condropatía/chondropathie rotuliana" = present (patellofemoral only).
   - "Trochlear/patellar arthrosis" = present.
   - "OA" without mentioning patella or patellofemoral compartment = NOT PF OA.
   - Report of general "gonartrosis" or "Gonarthrose" without patellar reference = not_stated.

8. Effusion: Pathological joint effusion. Rules:
   - "Joint effusion", "synovial effusion", "articular effusion", "hydrops" = present.
   - "Small effusion", "moderate effusion", "large effusion" = present.
   - ABSENT examples (do NOT mark present): "trace fluid", "minimal physiological fluid", "physiological amount of fluid", "tiny amount of synovial fluid", "no effusion", "ohne Erguss", "bez izljeva", "bez slobodne tekucine", "efüzyon izlenmedi", "efüzyon yok", "sin derrame", "geen hydrops", "geen effusie", "pas d'epanchement", "bez výpotku", "bez artritidy".
   - "Minimal fluid / trace fluid without clinical significance" = absent.
   - CRITICAL: Do NOT mark effusion as present simply because effusion is mentioned as negated. Read the full sentence carefully.
   - When effusion IS present with a clear quote, always provide the verbatim positive statement as exact_quote.

9. Synovitis: Synovial thickening, synovitis, pannus, or villonodular proliferation, ONLY when EXPLICITLY mentioned. Rules:
   - Joint effusion ALONE does NOT imply synovitis (even large effusions can be simple).
   - "Effusion with synovial thickening" or "effusion compatible with synovitis" = present.
   - ABSENT examples: "simple effusion", "effusion without synovitis", "bez sinovitisa", "sinovit yok", "sin signos de sinovitis", "geen synovitis", "ohne Synovitis", "no synovial thickening", "unremarkable synovium", "synovial fold not thickened".
   - If the report mentions effusion but does NOT mention synovial thickening/synovitis, mark synovitis as not_stated (not as present).
   - If a report says "small effusion" without any synovial mention = not_stated for Synovitis.

10. Baker's: Baker's cyst, popliteal cyst, gastrocnemius-semimembranosus bursal distension. Rules:
    - Any explicitly named Baker's/popliteal cyst of clinical size = present.
    - "Tiny/trace/minimal popliteal bursal fluid" or "small amount of fluid in the gastrocnemius-semimembranosus bursa without discrete cyst formation" = absent.
    - "No popliteal cyst", "no Baker's cyst", "geen Bakercyste", "Bakerzyste nicht nachweisbar", "bez Bakerove ciste", "sin quiste popliteo" = absent.
    - Gastrocnemius-semimembranosus bursa with measurable dimensions (e.g., "popliteal cyst 21 x 17 mm") = present.

11. Contusion: Traumatic bone bruise, bone contusion, or traumatic subchondral bone marrow edema. Rules:
    - Must be EXPLICITLY TRAUMATIC or described as bone bruise/contusion/marrow edema.
    - "Bone marrow edema", "kostani edem", "kemik ödemi", "marrow signal change", "osseous contusion" = present ONLY with traumatic context.
    - CRITICAL DISTINCTION - these are NOT bone contusion:
      * Soft tissue edema (edem mekih tkiva, yumuşak doku ödemi)
      * Peri-ligamentous edema (edema around ACL, MCL, capsule)
      * Subcutaneous edema
      * Subchondral sclerosis or cysts (these are OA, not contusion)
      * Subchondral bone marrow changes in the context of OA (degenerative, not traumatic)
    - "Edem" (Croatian) alone WITHOUT "kostani/kosti" qualifier = NOT contusion.
    - "Ödem" (Turkish) alone WITHOUT "kemik/kemiği" qualifier = NOT contusion.
    - "Bone bruise", "Knochenkontusion/Knochenödem" (German), "contusion osseuse" (French), "contusión ósea" (Spanish), "botcontusie" (Dutch) = present.
    - "Trabecular injury", "marrow edema pattern" following acute trauma = present.

12. Fracture: Acute cortical bone disruption, avulsion fracture (Segond, tibial spine, fibular head), depressed tibial plateau fracture. Rules:
    - Old, healed, or chronic fractures = absent.
    - "Old fracture", "healed fracture", "prior fracture", "known fracture" with no acute component = absent.
    - "Stress fracture" with active marrow edema = present.
    - Osteophytes, subchondral cysts, or bone spurs = NOT fractures.
    - "Avulsion fracture" unless specified as "old/healed" = present.
    - "Fissure", "crack", "cortical break" = present.

ANTI-HALLUCINATION & EXTRACTION RULES:
1. Output MUST be valid JSON containing ALL 12 KEYS.
2. "reasoning": Concise clinical justification (max 25 words). Must explain WHY you chose present/absent/not_stated.
3. "exact_quote": Verbatim word-for-word copy from REPORT. MINIMUM 3 WORDS required for "present" or "absent" state. If genuinely not mentioned, output "None" and use "not_stated".
4. "state": EXACTLY ONE of ["present", "absent", "not_stated"]:
   - "present": Unambiguous, explicit, positive clinical finding. Requires a verbatim supporting quote.
   - "absent": Explicit statement the structure is normal, intact, or negated. Requires verbatim negation quote.
   - "not_stated": Structure was NOT evaluated or mentioned. Use this LIBERALLY when unsure.
   - NEVER mark "absent" if not mentioned. NEVER mark "present" based on weak/ambiguous evidence.
5. "confidence": EXACTLY ONE of ["high", "medium", "low"]. Use "low" for hedged language, "medium" for inferred findings.
6. SHARED MENISCAL FINDINGS: "Menisci intact/normal/unremarkable" = BOTH absent. "Bilateral meniscal tears" = BOTH present.
7. COMPARTMENT DISCIPLINE: A single OA finding in one compartment does NOT automatically apply to others. Read each compartment description separately.
8. SPECIFICITY BIAS: When uncertain between "present" vs "not_stated", choose "not_stated". When uncertain between "absent" vs "not_stated", choose "not_stated". False positives are penalized more than false negatives.

MULTILINGUAL NEGATION REFERENCE (common absent-state phrases):
  English absent: "no [X]", "without [X]", "not seen", "intact", "unremarkable", "within normal limits", "no evidence of"
  Croatian/Serbian absent: "uredno", "intaktno", "bez rupture", "bez izljeva", "bez sinovitisa", "bez nalaza", "bez Bakerove ciste", "uredan nalaz", "bez patologije"
  Turkish absent: "normal", "intakt", "yok", "izlenmedi", "saptanmadi", "görülmedi", "saglamdır" — e.g., "efüzyon izlenmedi" = no effusion
  German absent: "intakt", "regelrecht", "unauffällig", "kein Erguss", "kein Einriss", "ohne Befund", "Bakerzyste nicht nachweisbar"
  Spanish absent: "normal", "íntegro/a", "intacto/a", "sin rotura", "sin derrame", "sin sinovitis", "no se observa", "conservado"
  Dutch absent: "gaaf", "intact", "geen hydrops", "geen scheur", "geen Bakercyste", "niet aangetoond"
  French absent: "intact", "normal", "sans anomalie", "pas d'épanchement", "sans synovite", "pas de kyste"
  Greek absent: "φυσιολογικό", "ακέραιο", "χωρίς", "δεν διαπιστώθηκε"
  Russian/Bulgarian absent: "нет разрыва", "нет выпота", "не изменён", "интактный"

EXAMPLES:

EXAMPLE 1 (English, ACL torn + medial meniscus + effusion with synovitis):
Report: "Anterior cruciate ligament is completely torn. Medial meniscus shows a complex tear at the posterior horn. Lateral meniscus is unremarkable. No MCL injury. Moderate joint effusion with synovial thickening. No popliteal cyst. No bone contusion. No fracture. Cartilage intact bilaterally."
Output:
{{
  "ACL": {{"reasoning": "Explicit complete tear stated.", "exact_quote": "Anterior cruciate ligament is completely torn.", "state": "present", "confidence": "high"}},
  "MCL": {{"reasoning": "Explicitly negated - no MCL injury.", "exact_quote": "No MCL injury.", "state": "absent", "confidence": "high"}},
  "Medial Meniscus": {{"reasoning": "Complex posterior horn tear explicitly stated.", "exact_quote": "Medial meniscus shows a complex tear at the posterior horn.", "state": "present", "confidence": "high"}},
  "Lateral Meniscus": {{"reasoning": "Explicitly unremarkable.", "exact_quote": "Lateral meniscus is unremarkable.", "state": "absent", "confidence": "high"}},
  "Medial OA": {{"reasoning": "Cartilage intact bilaterally; no medial compartment OA stated.", "exact_quote": "Cartilage intact bilaterally.", "state": "absent", "confidence": "high"}},
  "Lateral OA": {{"reasoning": "Cartilage intact bilaterally; no lateral OA.", "exact_quote": "Cartilage intact bilaterally.", "state": "absent", "confidence": "high"}},
  "PF OA": {{"reasoning": "Cartilage intact bilaterally; no PF OA.", "exact_quote": "Cartilage intact bilaterally.", "state": "absent", "confidence": "high"}},
  "Effusion": {{"reasoning": "Moderate joint effusion explicitly stated.", "exact_quote": "Moderate joint effusion with synovial thickening.", "state": "present", "confidence": "high"}},
  "Synovitis": {{"reasoning": "Synovial thickening explicitly co-stated with effusion.", "exact_quote": "Moderate joint effusion with synovial thickening.", "state": "present", "confidence": "high"}},
  "Baker's": {{"reasoning": "Explicitly negated - no popliteal cyst.", "exact_quote": "No popliteal cyst.", "state": "absent", "confidence": "high"}},
  "Contusion": {{"reasoning": "Explicitly negated - no bone contusion.", "exact_quote": "No bone contusion.", "state": "absent", "confidence": "high"}},
  "Fracture": {{"reasoning": "Explicitly negated - no fracture.", "exact_quote": "No fracture.", "state": "absent", "confidence": "high"}}
}}

EXAMPLE 2 (Croatian, normal report with incidental physiological fluid):
Report: "Prednji križni ligament je intaktan i normalnog toka. Stražnji križni ligament intaktan. Lateralni i medijalni kolateralni ligamenti uredni. Medijalni menisk urednoga nalaza. Lateralni menisk urednoga nalaza. Hondromalacija patele II stupnja. Minimalna fiziološka količina tekućine u zglobu. Nema Baker-ove ciste. Bez koštanih kontuzija."
Output:
{{
  "ACL": {{"reasoning": "Explicitly stated intact and normal course in Croatian.", "exact_quote": "Prednji križni ligament je intaktan i normalnog toka.", "state": "absent", "confidence": "high"}},
  "MCL": {{"reasoning": "Lateral and medial collateral ligaments explicitly normal (uredni).", "exact_quote": "Lateralni i medijalni kolateralni ligamenti uredni.", "state": "absent", "confidence": "high"}},
  "Medial Meniscus": {{"reasoning": "Medial meniscus explicitly normal finding (urednog nalaza).", "exact_quote": "Medijalni menisk urednoga nalaza.", "state": "absent", "confidence": "high"}},
  "Lateral Meniscus": {{"reasoning": "Lateral meniscus explicitly normal finding.", "exact_quote": "Lateralni menisk urednoga nalaza.", "state": "absent", "confidence": "high"}},
  "Medial OA": {{"reasoning": "No medial compartment OA mentioned; only PF chondromalacia.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Lateral OA": {{"reasoning": "No lateral compartment OA mentioned.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "PF OA": {{"reasoning": "Grade II patellar chondromalacia explicitly stated.", "exact_quote": "Hondromalacija patele II stupnja.", "state": "present", "confidence": "high"}},
  "Effusion": {{"reasoning": "Minimal physiological fluid — NOT pathological effusion.", "exact_quote": "Minimalna fiziološka količina tekućine u zglobu.", "state": "absent", "confidence": "high"}},
  "Synovitis": {{"reasoning": "No synovial thickening mentioned; only physiological fluid.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Baker's": {{"reasoning": "Baker's cyst explicitly negated (Nema Baker-ove ciste).", "exact_quote": "Nema Baker-ove ciste.", "state": "absent", "confidence": "high"}},
  "Contusion": {{"reasoning": "Bone contusions explicitly negated (Bez koštanih kontuzija).", "exact_quote": "Bez koštanih kontuzija.", "state": "absent", "confidence": "high"}},
  "Fracture": {{"reasoning": "No fracture mentioned in report.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}}
}}

EXAMPLE 3 (Turkish, knee with MCL sprain + ACL + menisci intact):
Report: "SOL DİZ MRG. Bulgular: Ön çapraz bağ intakt. Arka çapraz bağ intakt. İç yan bağ (MCL) grade 2 yırtık, peri-ligamentöz ödem mevcut. Medial ve lateral menisküs posterior hornda grade II dejenerasyon. Efüzyon izlenmedi. Popliteal kist izlenmedi. Kemik ödemi saptanmadı."
Output:
{{
  "ACL": {{"reasoning": "Ön çapraz bağ (ACL) explicitly stated intact.", "exact_quote": "Ön çapraz bağ intakt.", "state": "absent", "confidence": "high"}},
  "MCL": {{"reasoning": "MCL grade 2 tear with peri-ligamentous edema explicitly stated.", "exact_quote": "İç yan bağ (MCL) grade 2 yırtık, peri-ligamentöz ödem mevcut.", "state": "present", "confidence": "high"}},
  "Medial Meniscus": {{"reasoning": "Grade II degeneration without articular extension — not a tear.", "exact_quote": "Medial ve lateral menisküs posterior hornda grade II dejenerasyon.", "state": "absent", "confidence": "high"}},
  "Lateral Meniscus": {{"reasoning": "Grade II degeneration without articular extension — not a tear.", "exact_quote": "Medial ve lateral menisküs posterior hornda grade II dejenerasyon.", "state": "absent", "confidence": "high"}},
  "Medial OA": {{"reasoning": "No medial compartment OA mentioned.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Lateral OA": {{"reasoning": "No lateral compartment OA mentioned.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "PF OA": {{"reasoning": "No patellofemoral OA mentioned.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Effusion": {{"reasoning": "Effusion explicitly not observed (izlenmedi).", "exact_quote": "Efüzyon izlenmedi.", "state": "absent", "confidence": "high"}},
  "Synovitis": {{"reasoning": "No synovial thickening mentioned; effusion absent.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Baker's": {{"reasoning": "Popliteal cyst explicitly not observed.", "exact_quote": "Popliteal kist izlenmedi.", "state": "absent", "confidence": "high"}},
  "Contusion": {{"reasoning": "Bone edema explicitly not found (kemik ödemi saptanmadı).", "exact_quote": "Kemik ödemi saptanmadı.", "state": "absent", "confidence": "high"}},
  "Fracture": {{"reasoning": "No fracture mentioned in report.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}}
}}

EXAMPLE 4 (Spanish, multi-compartment OA + ACL + effusion, NO synovitis):
Report: "Técnica: RMN de la rodilla. Hallazgos: Rotura del LCA. Menisco interno con desgarro en asta posterior. Menisco externo íntegro. Artrosis femorotibial medial con pinzamiento del espacio articular. Sin artrosis lateral ni patelofemoral. Derrame articular moderado. Sin engrosamiento sinovial. Sin quiste poplíteo. Contusiones óseas femorotibiales mediales. Sin fractura."
Output:
{{
  "ACL": {{"reasoning": "Explicit ACL tear (Rotura del LCA) stated.", "exact_quote": "Rotura del LCA.", "state": "present", "confidence": "high"}},
  "MCL": {{"reasoning": "No MCL/LCM mention in report.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Medial Meniscus": {{"reasoning": "Medial meniscus (menisco interno) posterior horn tear stated.", "exact_quote": "Menisco interno con desgarro en asta posterior.", "state": "present", "confidence": "high"}},
  "Lateral Meniscus": {{"reasoning": "Lateral meniscus (menisco externo) explicitly intact.", "exact_quote": "Menisco externo íntegro.", "state": "absent", "confidence": "high"}},
  "Medial OA": {{"reasoning": "Medial femorotibial arthrosis with joint space narrowing explicitly stated.", "exact_quote": "Artrosis femorotibial medial con pinzamiento del espacio articular.", "state": "present", "confidence": "high"}},
  "Lateral OA": {{"reasoning": "Lateral arthrosis explicitly negated.", "exact_quote": "Sin artrosis lateral ni patelofemoral.", "state": "absent", "confidence": "high"}},
  "PF OA": {{"reasoning": "Patellofemoral arthrosis explicitly negated (ni patelofemoral).", "exact_quote": "Sin artrosis lateral ni patelofemoral.", "state": "absent", "confidence": "high"}},
  "Effusion": {{"reasoning": "Moderate articular effusion explicitly stated.", "exact_quote": "Derrame articular moderado.", "state": "present", "confidence": "high"}},
  "Synovitis": {{"reasoning": "Synovial thickening explicitly negated (Sin engrosamiento sinovial).", "exact_quote": "Sin engrosamiento sinovial.", "state": "absent", "confidence": "high"}},
  "Baker's": {{"reasoning": "Popliteal cyst explicitly negated (Sin quiste poplíteo).", "exact_quote": "Sin quiste poplíteo.", "state": "absent", "confidence": "high"}},
  "Contusion": {{"reasoning": "Medial femorotibial bone contusions explicitly stated.", "exact_quote": "Contusiones óseas femorotibiales mediales.", "state": "present", "confidence": "high"}},
  "Fracture": {{"reasoning": "Fracture explicitly negated (Sin fractura).", "exact_quote": "Sin fractura.", "state": "absent", "confidence": "high"}}
}}

EXAMPLE 5 (English minimal report, physiological fluid, specificity test):
Report: "Anterior cruciate ligament is normal. Menisci are unremarkable. Minimal physiological joint fluid."
Output:
{{
  "ACL": {{"reasoning": "ACL explicitly stated as normal.", "exact_quote": "Anterior cruciate ligament is normal.", "state": "absent", "confidence": "high"}},
  "MCL": {{"reasoning": "Not addressed in report.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Medial Meniscus": {{"reasoning": "Menisci stated as unremarkable — applies to both.", "exact_quote": "Menisci are unremarkable.", "state": "absent", "confidence": "high"}},
  "Lateral Meniscus": {{"reasoning": "Menisci stated as unremarkable — applies to both.", "exact_quote": "Menisci are unremarkable.", "state": "absent", "confidence": "high"}},
  "Medial OA": {{"reasoning": "Not addressed.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Lateral OA": {{"reasoning": "Not addressed.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "PF OA": {{"reasoning": "Not addressed.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Effusion": {{"reasoning": "Minimal physiological fluid — NOT pathological effusion.", "exact_quote": "Minimal physiological joint fluid.", "state": "absent", "confidence": "high"}},
  "Synovitis": {{"reasoning": "Not addressed; only physiological fluid present.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Baker's": {{"reasoning": "Not addressed.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Contusion": {{"reasoning": "Not addressed.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}},
  "Fracture": {{"reasoning": "Not addressed.", "exact_quote": "None", "state": "not_stated", "confidence": "high"}}
}}

REPORT:
{report}
"""

def clean_txt(s: str) -> str:
    # Unicode-aware stripping keeps Greek, Spanish, German, French, Dutch letters
    return re.sub(r"[^\w]", "", str(s).lower(), flags=re.UNICODE)


def parse_json_response(raw_text: str, uid: str, original_report: str = "") -> dict | None:
    """Safely extracts JSON from the LLM output with a Quadruple-Layer Clinical Hallucination Shield."""
    if not raw_text or not isinstance(raw_text, str):
        print(f"[ERROR] Empty raw_text for {uid}")
        return None
        
    clean_json = None
    md_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL | re.IGNORECASE)
    if md_match:
        clean_json = md_match.group(1).strip()
    else:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            clean_json = raw_text[start:end+1].strip()
            
    if not clean_json:
        print(f"[ERROR] No JSON block found in output for {uid}")
        return None
        
    try:
        clean_json_sanitized = re.sub(r",\s*([}\]])", r"\1", clean_json)
        try:
            data = json.loads(clean_json_sanitized)
        except json.JSONDecodeError:
            try:
                ast_str = clean_json_sanitized.replace("true", "True").replace("false", "False").replace("null", "None")
                data = ast.literal_eval(ast_str)
            except Exception:
                data = None
            
        out = {"StudyInstanceUID": str(uid).strip()}
        
        def normalize_key(k):
            return re.sub(r"[^a-zA-Z0-9]", "", str(k)).lower()
            
        target_map = {normalize_key(t): t for t in TARGETS}
        
        normalized_data = {}
        if data is not None:
            for k, val_item in data.items():
                norm_k = normalize_key(k)
                if norm_k in target_map:
                    normalized_data[target_map[norm_k]] = val_item
        else:
            # SOTA Fallback: Regex parsing if JSON/AST completely fails
            for t in TARGETS:
                escaped_t = re.escape(t)
                pattern = r'[\x22\x27]?' + escaped_t + r'[\x22\x27]?\s*:\s*\{([^}]+)\}'
                match = re.search(pattern, raw_text, re.IGNORECASE)
                if match:
                    inner = match.group(1)
                    state_match = re.search(r'[\x22\x27]?state[\x22\x27]?\s*:\s*[\x22\x27]([^\x22\x27]+)[\x22\x27]', inner, re.IGNORECASE)
                    quote_match = re.search(r'[\x22\x27]?exact_quote[\x22\x27]?\s*:\s*[\x22\x27]([^\x22\x27]*)[\x22\x27]', inner, re.IGNORECASE)
                    conf_match = re.search(r'[\x22\x27]?confidence[\x22\x27]?\s*:\s*[\x22\x27]([^\x22\x27]*)[\x22\x27]', inner, re.IGNORECASE)
                    if state_match:
                        normalized_data[t] = {
                            "state": state_match.group(1).strip(),
                            "exact_quote": quote_match.group(1).strip() if quote_match else "",
                            "confidence": conf_match.group(1).strip() if conf_match else "high",
                        }
            
        found_targets = sum(1 for t in TARGETS if t in normalized_data)
        if found_targets < 6:
            return None

        for t in TARGETS:
            val = normalized_data.get(t, {})
            raw_state = val.get("state") if isinstance(val, dict) else None
            exact_quote = str(val.get("exact_quote", "")).strip() if isinstance(val, dict) else ""
            
            is_present = False
            is_absent = False
            
            if isinstance(raw_state, bool):
                is_present = raw_state
                is_absent = not raw_state
            elif isinstance(raw_state, (int, float)):
                is_present = (raw_state == 1)
                is_absent = (raw_state == 0)
            elif isinstance(raw_state, str):
                s_low = raw_state.lower().strip(" .,'\"\n")
                # Priority 1: Check for explicit not_stated / none / missing
                if any(ns in s_low for ns in ["not_stated", "not stated", "none", "unknown", "unclear", "missing", "n/a"]):
                    is_present = False
                    is_absent = False
                # Priority 2: Exact matching for single status tokens
                elif s_low in [
                    "absent", "normal", "intact", "unremarkable", "negative", "negativo", 
                    "negatief", "no", "ausente", "afwezig", "unauffällig", "unauffaellig", 
                    "regelrecht", "intakt", "conservado", "conservada", "íntegro", "integro", "χωρίς"
                ]:
                    is_absent = True
                elif s_low in [
                    "present", "torn", "tear", "fracture", "positive", "positivo", 
                    "positief", "presente", "vorhanden", "anwesend"
                ]:
                    is_present = True
                # Priority 3: Negation-safe phrase matching
                else:
                    absent_phrases = [
                        "no tear", "no fracture", "no effusion", "not present", "not seen", "without tear", 
                        "within normal limits", "sin rotura", "geen scheur", "keine ruptur", "ohne befund"
                    ]
                    present_phrases = [
                        "present", "torn", "tear", "fracture", "positive", "mild", "moderate", "severe", "abnormal"
                    ]
                    negation_prefixes = [
                        "no ", "not ", "without ", "free of ", "kein", "sin ", "geen ", "pas de ", "negative", "ruled out"
                    ]
                    # Never classify finding as positive if accompanied by negation
                    if any(x in s_low for x in absent_phrases) or any(neg in s_low for neg in negation_prefixes):
                        is_absent = True
                    elif any(x in s_low for x in present_phrases):
                        is_present = True
            
            # =========================================================================
            # CLINICAL HALLUCINATION SHIELD
            # =========================================================================
            q_low = exact_quote.lower().strip()
            # Shield 1: Discard ungrounded labels with empty or "None" quote
            # For "present" state: always require a non-trivial verbatim quote
            # For "absent" state: allow multilingual negation triggers even without explicit quote
            NONE_EQUIVALENTS = {"none", "null", "n/a", "", "not mentioned", "not stated", 
                                  "none.", "not applicable", "not found", "n.a.", "na"}
            MIN_PRESENT_QUOTE_WORDS = 3  # present claims need at least 3-word quotes
            
            if q_low in NONE_EQUIVALENTS:
                is_present = False  # Always nullify ungrounded present claims
                # Only nullify absent if we have no in-report evidence
                if is_absent:
                    # Check if report itself contains negation for this target
                    target_negations = {
                        "ACL": ["acl intact", "acl normal", "intaktan", "intakt", "acl is intact", "normal course",
                                "ön çapraz bağ intakt", "prednji križni intaktan"],
                        "MCL": ["mcl intact", "no mcl", "collateral ligaments intact", "ligamenti uredni",
                                "kolateralni uredni", "kolateralni ligamenti uredni"],
                        "Effusion": ["no effusion", "without effusion", "bez izljeva", "efüzyon yok",
                                     "efüzyon izlenmedi", "sin derrame", "geen hydrops", "kein erguss",
                                     "pas d'epanchement", "bez výpotku", "bez artritidy",
                                     "no joint effusion", "without joint effusion"],
                        "Synovitis": ["no synovitis", "bez sinovitisa", "sinovit yok", "sin sinovitis",
                                      "geen synovitis", "no synovial thickening", "ohne synovitis",
                                      "simple effusion", "effusion without synovitis"],
                        "Baker's": ["no baker", "no popliteal", "nema baker", "popliteal cyst izlenmedi",
                                    "geen bakercyste", "bakerzyste nicht", "sin quiste popliteo",
                                    "bez bakerove ciste"],
                        "Contusion": ["no contusion", "no bone bruise", "no bone marrow", "bez kontuzije",
                                      "kontüzyon yok", "kemik ödemi saptanmadı", "keine knochenkontusion",
                                      "sin contusion osea"],
                        "Fracture": ["no fracture", "bez frakture", "kırık yok", "keine fraktur",
                                     "sin fractura", "geen fractuur", "sans fracture"],
                        "Medial Meniscus": ["menisci intact", "menisci unremarkable", "menisci normal",
                                             "menisk uredan", "meniscus intact", "menisci are intact"],
                        "Lateral Meniscus": ["menisci intact", "menisci unremarkable", "menisci normal",
                                              "menisk uredan", "meniscus intact", "menisci are intact"],
                        "Medial OA": ["no medial oa", "no medial arthrosis", "no medial arthritis",
                                       "no medial compartment oa", "cartilage intact"],
                        "Lateral OA": ["no lateral oa", "no lateral arthrosis", "cartilage intact"],
                        "PF OA": ["no patellofemoral", "no pf oa", "patellofemoral normal", "cartilage intact"],
                    }
                    neg_terms = target_negations.get(t, [])
                    has_in_report_negation = any(neg in r_lower for neg in neg_terms)
                    if not has_in_report_negation:
                        is_absent = False  # No negation found → truly not_stated
                    # If in-report negation found, keep is_absent = True (good absence evidence)
            elif is_present and len(q_low.split()) < MIN_PRESENT_QUOTE_WORDS:
                # Very short quote for a "present" claim is suspicious — downgrade weight
                if not any(q_low.startswith(w) for w in ["tear", "torn", "ruptur", "fracture"]):
                    is_present = False  # Reject ultra-short present quotes without clear tear terms
                
            # Shield 5: Grounding Verification Against Original Report
            # For "present" labels: strict grounding required (high FP rate otherwise)
            # For "absent" labels: more lenient (LLM paraphrases negation phrases)
            if (is_present or is_absent) and original_report:
                clean_q = clean_txt(exact_quote)
                clean_rep = clean_txt(original_report)
                if len(clean_q) > 3 and clean_q not in clean_rep:
                    q_words = set(re.findall(r"\b\w{4,}\b", q_low, flags=re.UNICODE))
                    rep_words = set(re.findall(r"\b\w{4,}\b", original_report.lower(), flags=re.UNICODE))
                    overlap = len(q_words & rep_words) / max(1, len(q_words))
                    
                    common_en = {"the", "and", "with", "knee", "tear", "intact", "effusion", 
                                  "ligament", "meniscus", "fluid", "normal", "unremarkable"}
                    is_english_report = len(common_en & rep_words) >= 2
                    
                    # Stricter for "present" claims; more lenient for "absent" negation claims
                    if is_present:
                        # English: require 30% overlap; non-English: require 20% overlap
                        threshold = 0.30 if is_english_report else 0.20
                    else:  # is_absent
                        # Absent: more lenient because LLM may paraphrase negations
                        threshold = 0.20 if is_english_report else 0.12
                    
                    if len(q_words) >= 3 and overlap < threshold:
                        if is_present:
                            is_present = False
                        else:
                            # For absent, check report for negation terms before nullifying
                            common_negations = ["no", "not", "without", "intact", "normal", "unremarkable",
                                                "absent", "bez", "yok", "izlenmedi", "kein", "sin", "geen",
                                                "uredno", "intaktno", "intakt", "intacto", "intakt"]
                            has_negation = any(neg in r_lower for neg in common_negations)
                            if not has_negation:
                                is_absent = False
            
            # Map verified findings to labels & confidence weights
            conf_str = str(val.get("confidence", "")).lower() if isinstance(val, dict) else ""
            if is_present:
                if "high" in conf_str:
                    out[t], out[f"{t}_weight"] = 0.95, 1.0
                elif "low" in conf_str:
                    out[t], out[f"{t}_weight"] = 0.65, 0.5
                else:
                    out[t], out[f"{t}_weight"] = 0.85, 0.85
            elif is_absent:
                if "high" in conf_str:
                    out[t], out[f"{t}_weight"] = 0.05, 1.0
                elif "low" in conf_str:
                    out[t], out[f"{t}_weight"] = 0.35, 0.5
                else:
                    out[t], out[f"{t}_weight"] = 0.15, 0.85
            else:
                # not_stated / hedged / missing
                if t in ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Effusion"]:
                    out[t], out[f"{t}_weight"] = 0.0, 0.0  # Strict Mask
                else:
                    out[t], out[f"{t}_weight"] = -1.0, 0.1  # Soft Negative Marker
            
            # =========================================================================
            # CLINICAL CALIBRATION LAYER (v4): Post-mapping per-target corrections
            # =========================================================================
            q_lower = exact_quote.lower().strip()
            r_lower = (original_report or "").lower()
            
            # CAL-1: Effusion trace/physiological downgrade
            # The model may call "small effusion" present but gold labels trace fluid as absent
            if t == "Effusion" and out.get(t, -999) >= 0.5:
                trace_effusion_patterns = [
                    "physiolog", "trace fluid", "trace amount", "minimal fluid",
                    "tiny amount", "fiziološka", "fizyolojik", "fisiologica",
                    "physiologische", "fysiologisch", "минимал", "physiol",
                    "small amount of synovial", "small amount of joint fluid",
                    "a small amount of fluid", "tiny physiolog",
                ]
                is_trace = any(pat in q_lower for pat in trace_effusion_patterns)
                if is_trace:
                    out[t], out[f"{t}_weight"] = 0.15, 0.85  # Downgrade to absent-equivalent

            # CAL-2: Contusion soft-tissue edema false positive suppressor  
            if t == "Contusion" and out.get(t, -999) >= 0.5:
                # Soft tissue edema patterns that are NOT bone contusion
                soft_tissue_patterns = [
                    "soft tissue", "mekih tkiva", "peri-ligament", "periligament",
                    "subcutaneous", "potkožni", "kapsula", "capsule edema",
                    "capsular edema", "yumuşak doku", "subkutan", "surrounding soft",
                    "periarticular soft", "peri-articular edema",
                ]
                bone_required_patterns = [
                    "bone", "kost", "kemik", "marrow", "osseous", "subchondral",
                    "trabecular", "contusion", "kontuzija", "kontüzyon",
                    "knochenkontusion", "knochenödem", "botcontusie", "contusion",
                    "osseus", "óseo", "kostani", "koštani",
                ]
                has_soft_tissue_only = any(p in q_lower for p in soft_tissue_patterns)
                has_bone_context = any(p in q_lower for p in bone_required_patterns)
                if has_soft_tissue_only and not has_bone_context:
                    out[t], out[f"{t}_weight"] = 0.15, 0.85  # Downgrade: soft tissue not bone

            # CAL-3: MCL peri-ligamentous edema strict attribution check
            if t == "MCL" and out.get(t, -999) >= 0.5:
                # Must mention MCL/medial collateral explicitly
                mcl_specificity_terms = [
                    "mcl", "medial collateral", "ligamentum collaterale mediale",
                    "inneres seitenband", "innerband", "unutarnji kolateralni",
                    "iç yan bağ", "bağ (mcl)", "lig. collaterale mediale",
                    "collaterale med", "med. collateral", "медиальная боковая",
                    "medijalni kolateralni", "ligamiento colateral medial",
                    "ligament collateral interne",
                ]
                mcl_mentioned = any(term in q_lower for term in mcl_specificity_terms)
                mcl_in_report = any(term in r_lower for term in mcl_specificity_terms)
                if not mcl_mentioned and not mcl_in_report:
                    # No MCL-specific term found — downgrade to uncertain
                    out[t], out[f"{t}_weight"] = 0.35, 0.5  # Reduce confidence

            # CAL-4: OA compartment cross-contamination guard
            # If quoting general "OA/arthrosis" without compartment, reduce weight
            oa_targets = {"Medial OA": ["medial", "medijal", "tibiofemoraal medial", "femorotibial med",
                                         "compartiment med", "mediales kompartiment", "medial tibiofem",
                                         "compartimento medial"],
                          "Lateral OA": ["lateral", "femorotibial lat", "compartiment lat",
                                          "laterales kompartiment", "compartimento lateral", "tibiofemoraal lat"],
                          "PF OA": ["patell", "trochle", "pf", "patellofemoral", "retropatellar",
                                    "chondromalacia patell", "chondropat", "condropat"]}
            if t in oa_targets and out.get(t, -999) >= 0.5:
                expected_terms = oa_targets[t]
                compartment_confirmed = any(term in q_lower for term in expected_terms)
                # Check if quote mentions a DIFFERENT OA compartment
                other_terms = []
                for oa_t, terms in oa_targets.items():
                    if oa_t != t:
                        other_terms.extend(terms)
                only_other_compartment = (any(term in q_lower for term in other_terms) and 
                                           not compartment_confirmed)
                if only_other_compartment:
                    out[t], out[f"{t}_weight"] = 0.15, 0.85  # Wrong compartment cited
                elif not compartment_confirmed:
                    # Generic OA mention without compartment — reduce confidence
                    out[t], out[f"{t}_weight"] = min(0.65, out.get(t, 0.65)), 0.6
            
            # CAL-5: Baker's cyst size/significance threshold
            if t == "Baker's" and out.get(t, -999) >= 0.5:
                minimal_cyst_patterns = [
                    "trace", "tiny", "minimal amount", "small amount of fluid in",
                    "no discrete cyst", "without discrete cyst", "bursal fluid",
                    "trace fluid in", "small amount of bursal",
                ]
                is_minimal = any(pat in q_lower for pat in minimal_cyst_patterns)
                if is_minimal:
                    out[t], out[f"{t}_weight"] = 0.2, 0.75

            # CAL-6: Synovitis — if effusion stated but synovitis quote is vague/generic
            if t == "Synovitis" and out.get(t, -999) >= 0.5:
                # Must have explicit synovitis mention, not just effusion
                explicit_synovitis = [
                    "synovit", "sinovit", "synovial thickening", "synovijalna zadeb",
                    "villonodular", "pannus", "synovial proliferat", "sinovijalna promjen",
                    "engrosamiento sinovial", "sinoviale verdikking", "synoviale verdick",
                    "synovite", "sinovite", "épaississement synovial",
                ]
                has_synovitis_term = any(term in q_lower for term in explicit_synovitis)
                if not has_synovitis_term:
                    # Probably effusion context being misread as synovitis
                    out[t], out[f"{t}_weight"] = 0.35, 0.5  # Reduce to uncertain
            
            # CAL-7: Fracture — healed/old/known qualifier suppressor  
            if t == "Fracture" and out.get(t, -999) >= 0.5:
                old_fracture_patterns = [
                    "old fracture", "healed fracture", "prior fracture", "chronic fracture",
                    "known fracture", "old avulsion", "healed avulsion", "stara fraktura",
                    "alte fraktur", "fracture ancienne", "fractura antigua", "oude fractuur",
                ]
                is_old = any(pat in q_lower for pat in old_fracture_patterns)
                if is_old:
                    out[t], out[f"{t}_weight"] = 0.05, 1.0  # Old fracture = absent
                    
        return out
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON for {uid}: {e}")
        return None


def run_offline_extraction(data_root: str, out_csv: str, model_id: str = None):
    """Compatibility entry point using the provenance-checked, strict vLLM path."""
    return auto_complete_extraction(data_root, out_csv, model_id=model_id, engine="vllm")


def resolve_local_model_path(repo_id: str) -> str:
    """If repo_id corresponds to a downloaded local HF snapshot, return the local directory path."""
    if not repo_id or os.path.isdir(repo_id):
        return repo_id
    proj_root = globals().get("PROJECT_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    repo_folder = f"models--{repo_id.replace('/', '--')}"
    candidate_bases = [
        os.environ.get("HF_HOME"),
        os.path.expanduser("~/.cache/huggingface"),
        os.path.abspath(os.path.join(proj_root, "data", "hf_cache")),
        "/home/iedc_ai_dgx1/.cache/huggingface",
    ]
    for base in candidate_bases:
        if not base:
            continue
        hub_dir = base if base.endswith("hub") else os.path.join(base, "hub")
        snapshots_dir = os.path.join(hub_dir, repo_folder, "snapshots")
        if os.path.isdir(snapshots_dir):
            snaps = [
                os.path.join(snapshots_dir, s)
                for s in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, s))
            ]
            if snaps:
                valid_snaps = [s for s in snaps if os.path.exists(os.path.join(s, "config.json"))]
                if valid_snaps:
                    valid_snaps.sort(key=lambda s: os.path.getmtime(s), reverse=True)
                    return valid_snaps[0]
    return repo_id


def auto_complete_extraction(
    data_root: str,
    out_csv: str,
    model_id: str | None = None,
    engine: str = "vllm",
    force: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    evaluate: bool = False,
) -> tuple[str, dict]:
    """Unified Auto-Detection & Completion Engine for NLP Pseudo-Labels.
    
    1. Scans train.csv to identify studies with a report and at least one missing target.
    2. Inspects out_csv to determine already completed studies.
    3. Reuses rows only when the run manifest, source report hash, prompt, engine, and model match.
    4. Selects one engine for the run and never changes labeling methodology after a runtime failure.
    5. Saves rows and provenance checkpoints atomically.
    
    Returns:
        (out_csv_path, stats_dict)
    """
    print("=" * 80)
    print("PHASE 1: NLP PSEUDO-LABEL AUTO-DETECTION & COMPLETION")
    print("=" * 80)
    
    train_path = os.path.join(data_root, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found at {train_path}")
        
    train_df = pd.read_csv(train_path)
    for target in TARGETS:
        if target not in train_df.columns:
            train_df[target] = pd.NA
    if train_df["StudyInstanceUID"].isna().any():
        raise ValueError("train.csv contains a missing StudyInstanceUID")
    train_df["StudyInstanceUID"] = train_df["StudyInstanceUID"].astype(str).str.strip()
    if train_df["StudyInstanceUID"].eq("").any() or train_df["StudyInstanceUID"].duplicated().any():
        raise ValueError("train.csv must have non-empty, unique StudyInstanceUID values")
    gold_mask = train_df[TARGETS].notna().any(axis=1)
    fully_labeled_mask = train_df[TARGETS].notna().all(axis=1)
    
    report_col = "Report" if "Report" in train_df.columns else ("report" if "report" in train_df.columns else None)
    if report_col is None:
        raise KeyError("Could not find 'Report' or 'report' column in train.csv")
        
    reports = train_df[report_col].fillna("").astype(str)
    report_present = reports.str.strip().ne("")
    needed_df = train_df[report_present].copy() if evaluate else train_df[~fully_labeled_mask & report_present].copy()
    needed_df["_report_text"] = reports.loc[needed_df.index]
    total_needed = len(needed_df)
    needed_uids = set(needed_df["StudyInstanceUID"])
    report_hashes = dict(
        zip(
            needed_df["StudyInstanceUID"],
            needed_df["_report_text"].map(_report_sha256),
        )
    )
    
    print(
        f"[STATUS] Dataset Studies: {len(train_df)} total | "
        f"Gold-labeled: {int(gold_mask.sum())} | Fully labeled: {int(fully_labeled_mask.sum())} | "
        f"Reports requiring missing-target completion: {total_needed}"
    )
    
    selected_engine = engine.lower()
    requested_model = model_id if model_id else os.environ.get("LLM_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
    quantization = os.environ.get("NLP_QUANTIZATION", "none").lower()
    large_unquantized_model = (
        re.search(r"(?:^|[-_/])7[0-9]b(?:[-_/]|$)", requested_model.lower()) is not None
        and quantization not in {"fp8", "fp8_e4m3", "fp8_e5m2"}
    )
    if selected_engine != "vllm":
        if selected_engine == "rules" and callable(globals().get("extract_by_rules")):
            pass
        else:
            raise ValueError("Only 'vllm' engine is supported. Rules engine has been removed for accuracy.")
    if selected_engine == "vllm" and LLM is None:
        raise ImportError("vLLM is required but not installed.")
    if selected_engine == "vllm" and large_unquantized_model:
        print(
            "[WARNING] Auto-enabling FP8 quantization for 70B/72B model. Unquantized "
            "weights exceed DGX memory."
        )
        os.environ["NLP_QUANTIZATION"] = "fp8"
        quantization = "fp8"
        large_unquantized_model = False

    resolved_model = (
        requested_model if selected_engine == "vllm" else "clinical-rules-v1"
    )
    prompt_sha256 = hashlib.sha256(build_prompt("__REPORT_TEXT__").encode("utf-8")).hexdigest()
    train_sha256 = _sha256_file(train_path)
    contract = {
        "extractor_version": EXTRACTOR_VERSION,
        "train_csv_sha256": train_sha256,
        "engine": selected_engine,
        "model_id": resolved_model,
        "prompt_sha256": prompt_sha256,
        "required_studies": len(needed_uids),
    }
    manifest_path = f"{out_csv}.manifest.json"
    output_dir = os.path.dirname(os.path.abspath(out_csv))
    os.makedirs(output_dir, exist_ok=True)
    if force:
        _atomic_csv(_empty_extraction_frame(), out_csv)

    # Evaluation-tolerant provenance contract
    prov_check = {
        "extractor_version": EXTRACTOR_VERSION,
        "engine": selected_engine,
    }
    if not evaluate:
        prov_check["model_id"] = resolved_model
        prov_check["prompt_sha256"] = prompt_sha256

    # 1. Reuse existing extractions if valid under provenance
    existing_results = []
    done_uids = set()
    manifest_matches = False
    
    if os.path.exists(manifest_path) and not force:
        try:
            with open(manifest_path, encoding="utf-8") as stream:
                old_manifest = json.load(stream)
            manifest_matches = all(old_manifest.get(k) == v for k, v in contract.items())
        except Exception as e:
            print(f"[WARNING] Manifest check ({e}); checking extraction CSV directly.")

    if os.path.exists(out_csv) and not force:
        try:
            existing_df = pd.read_csv(out_csv)
            valid_existing = _validated_extraction_rows(
                existing_df,
                needed_uids,
                report_hashes if not evaluate else None,
                prov_check,
            )
            if len(valid_existing) > 0:
                done_uids = set(valid_existing["StudyInstanceUID"])
                existing_results = valid_existing.to_dict("records")
                print(
                    f"[AUTO-DETECT] Valid cached extractions: {len(done_uids)} / "
                    f"{total_needed}; incomplete or invalid rows will be regenerated."
                )
        except Exception as e:
            print(f"[WARNING] Could not parse cached extraction CSV ({e}). Starting fresh.")
            existing_results = []
            done_uids = set()
            
    # 2. Check for completion
    if len(done_uids) >= total_needed:
        final_cached_df = pd.DataFrame(existing_results) if existing_results else _empty_extraction_frame()
        _atomic_csv(final_cached_df, out_csv)
        _atomic_json(
            manifest_path,
            {
                **contract,
                "status": "complete",
                "completed_studies": len(done_uids),
                "pseudo_csv_sha256": _sha256_file(out_csv),
                "gold_studies": int(gold_mask.sum()),
                "fully_labeled_studies": int(fully_labeled_mask.sum()),
                "target_values_pending": int(train_df[TARGETS].isna().sum().sum()),
                "blank_report_studies": int((~report_present).sum()),
                "label_counts": {
                    target: {
                        "positive": int((final_cached_df[target] >= 0.5).sum()),
                        "negative": int(((final_cached_df[target] >= 0.0) & (final_cached_df[target] < 0.5)).sum()),
                        "masked": int((final_cached_df[target] < 0.0).sum()),
                    }
                    for target in TARGETS
                },
            },
        )
        print(f"[SUCCESS] All {len(done_uids)} requested studies already extracted and verified in {out_csv}.")
        return out_csv, {
            "status": "complete",
            "total": len(done_uids),
            "new": 0,
            "engine": selected_engine,
            "model_id": resolved_model,
        }
        
    remaining_df = needed_df[~needed_df["StudyInstanceUID"].isin(done_uids)].copy()
    print(f"[AUTO-DETECT] Remaining to extract: {len(remaining_df)} studies ({len(done_uids)/max(1, total_needed)*100:.1f}% previously done).")
    print(f"[CONFIG] NLP Extraction Engine Selected: '{selected_engine.upper()}'")
    print(f"[CONFIG] NLP source model/engine revision: {resolved_model}")
    
    start_time = time.time()
    _atomic_json(
        manifest_path,
        {**contract, "status": "in_progress", "completed_studies": len(done_uids)},
    )
        
    results = existing_results
    
    if selected_engine == "rules" and callable(globals().get("extract_by_rules")):
        rule_fn = globals()["extract_by_rules"]
        for _, row in remaining_df.iterrows():
            res = rule_fn(row["_report_text"], row["StudyInstanceUID"])
            if res:
                res.update(
                    report_sha256=_report_sha256(row["_report_text"]),
                    extractor_version=EXTRACTOR_VERSION,
                    engine=selected_engine,
                    model_id=resolved_model,
                    prompt_sha256=prompt_sha256,
                )
                results.append(res)
    elif selected_engine == "vllm":
        try:
            print(f"[INFO] Launching vLLM batch engine for {len(remaining_df)} studies...")
            local_resolved = resolve_local_model_path(requested_model)
            if local_resolved != requested_model:
                print(f"[CACHE] Resolved local snapshot for '{requested_model}' at: {local_resolved}")
                model_to_use = local_resolved
            else:
                model_to_use = requested_model
            requested_gpu_util = float(os.environ.get("VLLM_GPU_MEMORY_UTILIZATION", "0.85"))
            if not 0.0 < requested_gpu_util < 1.0:
                raise ValueError("VLLM_GPU_MEMORY_UTILIZATION must be between 0 and 1")
            total_gpu_gb = (
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            )
            memory = psutil.virtual_memory()
            used_gb = memory.used / (1024 ** 3)
            available_gb = memory.available / (1024 ** 3)
            safe_budget_gb = min(
                config.MEMORY_TARGET_GB - used_gb,
                available_gb - config.MIN_AVAILABLE_RAM_GB,
            )
            if safe_budget_gb <= 0:
                raise RuntimeError(
                    f"Insufficient unified-memory headroom for vLLM: {available_gb:.1f} GiB "
                    f"available; at least {config.MIN_AVAILABLE_RAM_GB:.1f} GiB must remain."
                )
            memory_target_util = min(0.99, safe_budget_gb / total_gpu_gb)
            gpu_util = min(requested_gpu_util, memory_target_util)
            if gpu_util < requested_gpu_util:
                print(
                    f"[SAFETY] Capping vLLM GPU memory utilization at {gpu_util:.3f} "
                    f"based on current use ({used_gb:.1f} GiB) and the "
                    f"{config.MEMORY_TARGET_GB:.0f} GiB target."
                )
            enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "0") in ["1", "true", "True"]
            use_quant = quantization

            llm_kwargs = {}
            if use_quant in ["fp8", "fp8_e4m3", "fp8_e5m2"]:
                llm_kwargs = {"quantization": "fp8"}
            elif use_quant in ["bitsandbytes", "bnb"]:
                print("[INFO] Note: bitsandbytes quantization is not supported in vLLM v1 engine. Running unquantized native precision.")
            elif use_quant not in ["none", "null", "false", "fp16", "bf16"]:
                llm_kwargs = {"quantization": use_quant}

            try:
                from transformers import AutoConfig, AutoTokenizer
                print(f"[PRE-FLIGHT] Verifying model config and tokenizer for '{model_to_use}'...")
                _cfg = AutoConfig.from_pretrained(model_to_use, trust_remote_code=True)
                _tok = AutoTokenizer.from_pretrained(model_to_use, trust_remote_code=True)
                print(f"[PRE-FLIGHT] Verified: model_type='{getattr(_cfg, 'model_type', 'unknown')}', tokenizer='{_tok.__class__.__name__}'.")
            except Exception as _pf_err:
                _hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
                print(f"[PRE-FLIGHT WARNING] Fast tokenizer/config check failed for '{model_to_use}': {_pf_err}")
                if not _hf_token and not os.path.exists(model_to_use):
                    print("[DIAGNOSTIC] HF_TOKEN is NOT set in environment or .env file!")
                    print("[DIAGNOSTIC] Unauthenticated HuggingFace requests on shared IPs frequently return HTTP 429/rate-limit error responses.")

            max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", str(DEFAULT_MAX_MODEL_LEN)))
            llm = LLM(
                model=model_to_use,
                enforce_eager=enforce_eager,
                max_model_len=max_model_len,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_util,
                trust_remote_code=True,
                **llm_kwargs
            )
            
            schema_dict = {
                "type": "object",
                "properties": {
                    t: {
                        "type": "object",
                        "properties": {
                            "reasoning": {"type": "string"},
                            "exact_quote": {"type": "string"},
                            "state": {"type": "string", "enum": ["present", "absent", "not_stated"]},
                            "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
                        },
                        "required": ["reasoning", "exact_quote", "state", "confidence"],
                        "additionalProperties": False
                    } for t in TARGETS
                },
                "required": TARGETS,
                "additionalProperties": False
            }
            schema_str = json.dumps(schema_dict)
            
            try:
                from vllm.sampling_params import GuidedDecodingParams
                guided = GuidedDecodingParams(json=schema_str)
                sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, guided_decoding=guided)
                decoding_mode = "GuidedDecodingParams"
            except Exception:
                try:
                    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens, guided_json=schema_str)
                    decoding_mode = "guided_json"
                except Exception:
                    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
                    decoding_mode = "unconstrained"
                    print("[INFO] Using unconstrained decoding at temperature 0.0 with deterministic clinical prompting.")

            global_failed_queue = []
            total_batches = ((len(remaining_df) - 1) // chunk_size) + 1
            for batch_idx, i in enumerate(range(0, len(remaining_df), chunk_size)):
                chunk_start = time.time()
                chunk = remaining_df.iloc[i:i+chunk_size]
                full_reports = chunk["_report_text"].astype(str).tolist()
                raw_reports = [report for report in full_reports]
                messages_chunk = [[{"role": "user", "content": build_prompt(r)}] for r in raw_reports]
                uids_chunk = chunk["StudyInstanceUID"].tolist()

                print(f"\n[BATCH {batch_idx + 1} / {total_batches}] Dispatching {len(chunk)} concurrent studies to vLLM engine...")
                outputs = llm.chat(messages_chunk, sampling_params, use_tqdm=True)
                chunk_elapsed = max(0.001, time.time() - chunk_start)
                
                new_success = 0
                for output, uid, report_str, full_report in zip(outputs, uids_chunk, raw_reports, full_reports):
                    text = output.outputs[0].text if (output.outputs and len(output.outputs) > 0) else ""
                    append_to_jsonl(uid, text, out_csv)
                    res = parse_json_response(text, uid, original_report=report_str)
                    if res:
                        res.update(
                            report_sha256=_report_sha256(full_report),
                            extractor_version=EXTRACTOR_VERSION,
                            engine=selected_engine,
                            model_id=resolved_model,
                            prompt_sha256=prompt_sha256,
                        )
                        results.append(res)
                        new_success += 1
                    else:
                        global_failed_queue.append((uid, report_str, full_report))

                df_out = pd.DataFrame(results).drop_duplicates(subset=["StudyInstanceUID"], keep="last")
                _atomic_csv(df_out, out_csv)
                saved_rows = _validated_extraction_rows(
                    df_out,
                    needed_uids,
                    report_hashes if not evaluate else None,
                    prov_check,
                )
                _atomic_json(
                    manifest_path,
                    {**contract, "status": "in_progress", "completed_studies": len(saved_rows)},
                )
                rps = len(chunk) / chunk_elapsed
                rem_studies = len(remaining_df) - (i + len(chunk))
                eta_s = rem_studies / rps if rps > 0 else 0
                print(
                    f"[THROUGHPUT] Batch {batch_idx + 1}: {len(chunk)} studies in {chunk_elapsed:.1f}s "
                    f"({rps:.1f} studies/sec) | Success: {new_success}/{len(chunk)} | "
                    f"Total Checkpointed: {len(df_out)} | ETA: {int(eta_s//60)}m {int(eta_s%60):02d}s"
                )
                
            # Global Retry Pass for Failed Reports
            max_global_retries = 3
            current_queue = global_failed_queue
            
            for attempt in range(max_global_retries):
                if not current_queue:
                    break
                    
                print(f"\n[INFO] Global Retry Pass {attempt + 1} / {max_global_retries} for {len(current_queue)} failed reports...")
                try:
                    from vllm.sampling_params import GuidedDecodingParams
                    guided_retry_p = GuidedDecodingParams(json=schema_str)
                    retry_params = SamplingParams(temperature=0.2, seed=42+attempt, max_tokens=max_tokens, guided_decoding=guided_retry_p)
                except Exception:
                    try:
                        retry_params = SamplingParams(temperature=0.2, seed=42+attempt, max_tokens=max_tokens, guided_json=schema_str)
                    except Exception:
                        retry_params = SamplingParams(temperature=0.2, seed=42+attempt, max_tokens=max_tokens)

                next_queue = []
                for i in range(0, len(current_queue), chunk_size):
                    chunk_items = current_queue[i:i+chunk_size]
                    retry_messages = [[{"role": "user", "content": build_prompt(item[1])}] for item in chunk_items]
                    retry_outputs = llm.chat(retry_messages, retry_params, use_tqdm=True)
                    
                    for output, (uid, report_str, full_report) in zip(retry_outputs, chunk_items):
                        text = output.outputs[0].text if (output.outputs and len(output.outputs) > 0) else ""
                        append_to_jsonl(uid, text, out_csv)
                        res = parse_json_response(text, uid, original_report=report_str)
                        if res:
                            res.update(
                                report_sha256=_report_sha256(full_report),
                                extractor_version=EXTRACTOR_VERSION,
                                engine=selected_engine,
                                model_id=resolved_model,
                                prompt_sha256=prompt_sha256,
                            )
                            results.append(res)
                        else:
                            next_queue.append((uid, report_str, full_report))
                            
                current_queue = next_queue
                df_out = pd.DataFrame(results).drop_duplicates(subset=["StudyInstanceUID"], keep="last")
                _atomic_csv(df_out, out_csv)
                saved_rows = _validated_extraction_rows(
                    df_out,
                    needed_uids,
                    report_hashes if not evaluate else None,
                    prov_check,
                )
                _atomic_json(manifest_path, {**contract, "status": "in_progress", "completed_studies": len(saved_rows)})
                print(f"[CHECKPOINT] Saved {len(df_out)} studies after retry pass {attempt + 1}.")
                
            if current_queue:
                print(f"[ERROR] {len(current_queue)} reports permanently failed after all global retries. Falling back to default empty weights.")
                for uid, _, full_report in current_queue:
                    fallback = {"StudyInstanceUID": uid}
                    for t in TARGETS:
                        fallback[t] = 0.0
                        fallback[f"{t}_weight"] = 0.0
                    fallback.update(
                        report_sha256=_report_sha256(full_report),
                        extractor_version=EXTRACTOR_VERSION,
                        engine=selected_engine,
                        model_id=resolved_model,
                        prompt_sha256=prompt_sha256,
                    )
                    results.append(fallback)
                df_out = pd.DataFrame(results).drop_duplicates(subset=["StudyInstanceUID"], keep="last")
                _atomic_csv(df_out, out_csv)
                saved_rows = _validated_extraction_rows(
                    df_out,
                    needed_uids,
                    report_hashes if not evaluate else None,
                    prov_check,
                )

        except Exception as vllm_err:
            _atomic_json(
                manifest_path,
                {
                    **contract,
                    "status": "failed",
                    "completed_studies": len(done_uids),
                    "error": f"{type(vllm_err).__name__}: {vllm_err}",
                },
            )
            raise RuntimeError(
                "vLLM extraction failed. Partial rows were checkpointed; rerun with the same "
                "engine/model to resume, or explicitly start a separate rules-label run."
            ) from vllm_err
        finally:
            if "llm" in locals():
                del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    final_df = (
        pd.DataFrame(results).drop_duplicates(subset=["StudyInstanceUID"], keep="last")
        if results
        else _empty_extraction_frame()
    )
    final_df = _validated_extraction_rows(
        final_df,
        needed_uids,
        report_hashes if not evaluate else None,
        prov_check,
    )
    missing_uids = needed_uids - set(final_df["StudyInstanceUID"])
    if missing_uids:
        _atomic_json(
            manifest_path,
            {
                **contract,
                "status": "failed",
                "completed_studies": len(final_df),
                "missing_studies": len(missing_uids),
            },
        )
        raise RuntimeError(
            f"NLP extraction is incomplete: {len(missing_uids)} / {total_needed} "
            "required studies have no complete, in-range labels"
        )
    _atomic_csv(final_df, out_csv)
    _atomic_json(
        manifest_path,
        {
            **contract,
            "status": "complete",
            "completed_studies": len(final_df),
            "pseudo_csv_sha256": _sha256_file(out_csv),
            "gold_studies": int(gold_mask.sum()),
            "fully_labeled_studies": int(fully_labeled_mask.sum()),
            "target_values_pending": int(train_df[TARGETS].isna().sum().sum()),
            "blank_report_studies": int((~report_present).sum()),
            "label_counts": {
                target: {
                    "positive": int((final_df[target] >= 0.5).sum()),
                    "negative": int(((final_df[target] >= 0.0) & (final_df[target] < 0.5)).sum()),
                    "masked": int((final_df[target] < 0.0).sum()),
                }
                for target in TARGETS
            },
        },
    )
    elapsed = time.time() - start_time
    print(f"[SUCCESS] NLP extraction completed in {elapsed:.1f}s. Total valid studies in {out_csv}: {len(final_df)}.")
    return out_csv, {
        "status": "complete",
        "total": len(final_df),
        "new": len(remaining_df),
        "engine": selected_engine,
        "model_id": resolved_model,
    }


def detect_language(report: str) -> str:
    if not isinstance(report, str):
        return "English"
    r_lower = report.lower()
    
    def has_words(words):
        return any(re.search(rf"\b{w}\b", r_lower) for w in words)
        
    if has_words(["bulgular", "eklem", "bag", "yırtık", "yirtik", "menisküs", "mr"]):
        return "Turkish"
    if has_words(["nalaz", "tetive", "ligament", "prikazuje", "pregled", "struktura", "intaktna", "ruptura"]):
        return "Croatian"
    if has_words(["befund", "beurteilung", "kreuzband", "meniskus", "gelenk", "innenmeniskus"]):
        return "German"
    if has_words(["informe", "hallazgos", "conclusion", "rotura", "derrame", "menisco"]):
        return "Spanish"
    if has_words(["conclusion", "examen", "ligament", "croise", "menisque", "epanchement"]):
        return "French"
    if has_words(["verslag", "conclusie", "kruisband", "meniscus", "hydrops", "geen"]):
        return "Dutch"
    if re.search(r"[\u0400-\u04FF]", report):
        return "Russian/Bulgarian"
    if re.search(r"[\u0370-\u03FF]", report):
        return "Greek"
    return "English"


def calculate_clinical_metrics(y_true, y_pred, threshold=0.5) -> dict:
    """Calculates sensitivity, specificity, precision, F1, and disagreement counts."""
    y_true_b = (y_true >= threshold).astype(int)
    y_pred_b = (y_pred >= threshold).astype(int)
    
    tp = int(np.sum((y_true_b == 1) & (y_pred_b == 1)))
    fp = int(np.sum((y_true_b == 0) & (y_pred_b == 1)))
    fn = int(np.sum((y_true_b == 1) & (y_pred_b == 0)))
    tn = int(np.sum((y_true_b == 0) & (y_pred_b == 0)))
    
    sensitivity = tp / max(1, tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / max(1, tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / max(1, tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * (precision * sensitivity) / max(1e-6, precision + sensitivity)
    disagreements = fp + fn
    
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "sens": sensitivity, "spec": specificity, "prec": precision, "f1": f1,
        "disagreements": disagreements
    }


def run_gold_evaluation(
    data_root: str = "data",
    engine: str = "vllm",
    model_id: str | None = None,
    force: bool = False,
    show_errors: bool = True,
    temp_dir: str = "temp_gold_eval",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[dict, pd.DataFrame]:
    """Unified Gold Standard Evaluation Suite for RSNA Knee MSK NLP Extractor.
    
    Evaluates both:
    1. Stated Clinical Coverage & ROC-AUC (LLM comprehension on addressed findings)
    2. End-to-End Calibrated ROC-AUC (aligns with labels.py MNAR soft-negative prior)
    """
    data_root = resolve_data_root(data_root)
    train_path = os.path.join(data_root, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found at {train_path}")

    train_df = pd.read_csv(train_path)
    
    gold_df = train_df[train_df["ACL"].notna()].copy()
    gold_df["StudyInstanceUID"] = gold_df["StudyInstanceUID"].astype(str).str.strip()
    
    print("=" * 96)
    print(f"RSNA KNEE MSK RADIOLOGY EVALUATION - {len(gold_df)} GOLD STUDIES")
    print(f"Data Root: {data_root} | Engine: {engine.upper()} | Model: {model_id or 'Default'}")
    print("=" * 96)
    
    os.makedirs(temp_dir, exist_ok=True)
    gold_df.to_csv(os.path.join(temp_dir, "train.csv"), index=False)
    out_csv = os.path.join(temp_dir, "gold_extractions.csv")
    
    start_time = time.time()
    out_path, stats = auto_complete_extraction(
        data_root=temp_dir,
        out_csv=out_csv,
        model_id=model_id,
        engine=engine,
        force=force,
        chunk_size=chunk_size,
        evaluate=True
    )
    elapsed = time.time() - start_time
    extracted_df = pd.read_csv(out_path)
    extracted_df["StudyInstanceUID"] = extracted_df["StudyInstanceUID"].astype(str).str.strip()
    print(f"\n[TIMING] Extraction finished in {elapsed:.2f}s ({elapsed / max(1, len(gold_df)):.2f}s per report).")
    
    merged = pd.merge(gold_df, extracted_df, on="StudyInstanceUID", suffixes=("_true", "_pred"))
    merged["Language"] = merged["Report"].apply(detect_language)
    merged["token_length"] = merged["Report"].apply(lambda x: len(str(x)) / 4)
    
    print("\n[DATASET PROFILE]")
    print(f"Total Evaluated Studies: {len(merged)}")
    print("Report Languages Represented:")
    for lang, cnt in merged["Language"].value_counts().items():
        print(f"  - {lang:18s}: {cnt:2d} reports ({cnt/len(merged)*100:4.1f}%)")
    
    print("\n" + "=" * 96)
    print("CLINICAL COMPREHENSION & ROC-AUC MATRIX (ACROSS ALL 12 TARGETS)")
    print("=" * 96)
    print(f"{'Target':<18} | {'Cov%':<6} | {'Stated AUC':<10} | {'Calib AUC':<10} | {'Sens':<6} | {'Spec':<6} | {'Prec':<6} | {'F1':<6} | {'Disagreements'}")
    print("-" * 96)
    
    stated_aucs = []
    calibrated_aucs = []
    disagreement_records = []
    summary_metrics = {}
    
    for t in TARGETS:
        y_true = merged[f"{t}_true"].values
        y_pred_raw = merged[f"{t}_pred"].values
        y_weight = merged[f"{t}_weight"].values if f"{t}_weight" in merged.columns else np.ones_like(y_true)
        y_weight = np.nan_to_num(y_weight, nan=0.0)
        
        # Stated filter: structure was addressed in the report (weight > 0 and pred >= 0)
        stated_mask = (y_weight > 0.0) & (y_pred_raw >= 0.0) & np.isfinite(y_true)
        coverage_pct = (np.sum(stated_mask) / max(1, len(y_true))) * 100.0
        
        # 1. Stated AUC
        y_true_stated = y_true[stated_mask]
        y_pred_stated = y_pred_raw[stated_mask]
        stated_auc = float("nan")
        if len(set(y_true_stated)) > 1:
            try:
                stated_auc = roc_auc_score(y_true_stated, y_pred_stated)
                stated_aucs.append(stated_auc)
            except ValueError:
                pass
                
        # 2. Calibrated AUC (aligns with labels.py MNAR calibration for unstated findings)
        gold_valid = y_true[np.isfinite(y_true)]
        gold_prevalence = np.mean(gold_valid) if len(gold_valid) > 0 else 0.05
        calibrated_soft_neg = min(0.15, gold_prevalence * 0.8)
        
        y_pred_calibrated = y_pred_raw.copy()
        unstated_mask = (y_pred_raw < 0.0) | ((y_pred_raw == 0.0) & (y_weight == 0.0))
        y_pred_calibrated[unstated_mask] = calibrated_soft_neg
        
        calib_auc = float("nan")
        if len(set(y_true)) > 1:
            try:
                calib_auc = roc_auc_score(y_true, y_pred_calibrated)
                calibrated_aucs.append(calib_auc)
            except ValueError:
                pass
                
        # Clinical classification metrics on stated findings
        if len(y_true_stated) > 0:
            m = calculate_clinical_metrics(y_true_stated, y_pred_stated, threshold=0.5)
            s_auc_str = f"{stated_auc:.4f}" if np.isfinite(stated_auc) else "  N/A  "
            c_auc_str = f"{calib_auc:.4f}" if np.isfinite(calib_auc) else "  N/A  "
            print(
                f"{t:<18} | {coverage_pct:5.1f}% | {s_auc_str:<10} | {c_auc_str:<10} | "
                f"{m['sens']:5.3f} | {m['spec']:5.3f} | {m['prec']:5.3f} | {m['f1']:5.3f} | "
                f"{m['disagreements']:2d} (TP:{m['TP']} FP:{m['FP']} FN:{m['FN']} TN:{m['TN']})"
            )
            summary_metrics[t] = {**m, "coverage": coverage_pct, "stated_auc": stated_auc, "calib_auc": calib_auc}
        else:
            s_auc_str = "  N/A  "
            c_auc_str = f"{calib_auc:.4f}" if np.isfinite(calib_auc) else "  N/A  "
            print(f"{t:<18} | {coverage_pct:5.1f}% | {s_auc_str:<10} | {c_auc_str:<10} |   N/A  |   N/A  |   N/A  |   N/A  |  0")
            summary_metrics[t] = {"coverage": coverage_pct, "stated_auc": stated_auc, "calib_auc": calib_auc}
            
        # Record disagreements for forensic audit
        for idx, row in merged.iterrows():
            yt = row[f"{t}_true"]
            yp = row[f"{t}_pred"]
            yw = row[f"{t}_weight"] if f"{t}_weight" in row else 1.0
            if (yp >= 0.0) and (yw > 0.0) and pd.notna(yt):
                pred_bin = 1 if yp >= 0.5 else 0
                true_bin = int(yt)
                if pred_bin != true_bin:
                    disagreement_records.append({
                        "UID": row["StudyInstanceUID"],
                        "Target": t,
                        "True": true_bin,
                        "Pred": yp,
                        "Lang": row["Language"],
                        "Report": str(row["Report"])[:120] + "..."
                    })
                    
    print("-" * 96)
    macro_stated = np.nanmean(stated_aucs) if stated_aucs else float("nan")
    macro_calib = np.nanmean(calibrated_aucs) if calibrated_aucs else float("nan")
    print(f"{'MACRO ROC-AUC':<18} |        | {macro_stated:8.4f}   | {macro_calib:8.4f}   |")
    print("=" * 96)
    
    # Performance breakdown by language
    print("\n" + "=" * 80)
    print("PERFORMANCE BREAKDOWN BY REPORT LANGUAGE")
    print("=" * 80)
    print(f"{'Language':<18} | {'N':<4} | {'Avg Words':<10} | {'Macro Stated AUC':<18} | {'Macro Calib AUC'}")
    print("-" * 80)
    
    for lang in sorted(merged["Language"].unique()):
        lang_df = merged[merged["Language"] == lang]
        l_stated = []
        l_calib = []
        for t in TARGETS:
            y_t = lang_df[f"{t}_true"].values
            y_p = lang_df[f"{t}_pred"].values
            y_w = lang_df[f"{t}_weight"].values if f"{t}_weight" in lang_df.columns else np.ones_like(y_t)
            y_w = np.nan_to_num(y_w, nan=0.0)
            
            st_m = (y_w > 0.0) & (y_p >= 0.0) & np.isfinite(y_t)
            if len(set(y_t[st_m])) > 1:
                try:
                    l_stated.append(roc_auc_score(y_t[st_m], y_p[st_m]))
                except ValueError:
                    pass
                    
            if len(set(y_t[np.isfinite(y_t)])) > 1:
                y_p_c = y_p.copy()
                g_prev = np.mean(y_t[np.isfinite(y_t)]) if len(y_t) > 0 else 0.05
                u_m = (y_p < 0.0) | ((y_p == 0.0) & (y_w == 0.0))
                y_p_c[u_m] = min(0.15, g_prev * 0.8)
                try:
                    l_calib.append(roc_auc_score(y_t[np.isfinite(y_t)], y_p_c[np.isfinite(y_t)]))
                except ValueError:
                    pass
                    
        s_res = f"{np.nanmean(l_stated):.4f}" if l_stated else "   N/A   "
        c_res = f"{np.nanmean(l_calib):.4f}" if l_calib else "   N/A   "
        avg_w = np.mean(lang_df["Report"].astype(str).apply(lambda x: len(x.split())))
        print(f"{lang:<18} | {len(lang_df):2d}   | {avg_w:8.1f}   | {s_res:<18} | {c_res}")
        
    print("=" * 80)
    
    # Forensic disagreement audit
    if show_errors and disagreement_records:
        print(f"\n[FORENSIC AUDIT] {len(disagreement_records)} CLINICAL DISAGREEMENTS IDENTIFIED:")
        for i, r in enumerate(disagreement_records[:15]):
            print(f"  [{i+1:2d}] Target: {r['Target']:<16} | True: {r['True']} vs Pred: {r['Pred']:.2f} | Lang: {r['Lang']}")
            print(f"       UID:    {r['UID']}")
            print(f"       Report: {r['Report']}")
        if len(disagreement_records) > 15:
            print(f"  ... and {len(disagreement_records) - 15} more.")
    elif not disagreement_records:
        print("\n[FORENSIC AUDIT] 100% PERFECT CONCORDANCE! Zero disagreements on stated findings.")
        
    return {
        "macro_stated_auc": macro_stated,
        "macro_calib_auc": macro_calib,
        "per_target": summary_metrics,
        "disagreements": len(disagreement_records),
    }, merged


def main():
    parser = argparse.ArgumentParser(description="Unified NLP Pseudo-Label Extraction & Gold Evaluation Engine")
    parser.add_argument("--data_root", type=str, default=None, help="Path to raw dataset directory containing train.csv")
    parser.add_argument("--out_csv", type=str, default=None, help="Output path for extracted pseudo-labels CSV")
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm", "rules"], help="NLP Engine")
    parser.add_argument("--model", type=str, default=None, help="HuggingFace model ID or local directory")
    parser.add_argument("--force", action="store_true", help="Force re-extraction ignoring checkpoints")
    parser.add_argument("--chunk_size", "--batch_size", type=int, default=DEFAULT_CHUNK_SIZE, help="Batch/chunk size of concurrent reports processed in one go")
    parser.add_argument("--evaluate", action="store_true", help="Run comprehensive evaluation on gold-standard studies")
    parser.add_argument("--show_errors", action=argparse.BooleanOptionalAction, default=True, help="Display forensic disagreement audit")
    args = parser.parse_args()

    if args.evaluate:
        data_root = args.data_root or (os.path.join(PROJECT_ROOT, "data") if os.path.exists(os.path.join(PROJECT_ROOT, "data", "train.csv")) else "data")
        run_gold_evaluation(
            data_root=data_root,
            engine=args.engine,
            model_id=args.model,
            force=args.force,
            show_errors=args.show_errors,
            chunk_size=args.chunk_size,
        )
    else:
        data_root = resolve_data_root(args.data_root) if args.data_root else os.path.join(PROJECT_ROOT, "data")
        out_csv = args.out_csv or os.path.join(data_root, "pseudo_labels.csv")
        auto_complete_extraction(
            data_root=data_root,
            out_csv=out_csv,
            model_id=args.model,
            engine=args.engine,
            force=args.force,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
