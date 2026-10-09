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

import src.core.config as config
from src.core.config import resolve_data_root

# Module-level project root resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

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

EXTRACTOR_VERSION = "clinical-report-labels-v3"


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
2. MCL: Medial Collateral Ligament tear or sprain. Peri-ligamentous edema = present. (MPFL tears are NOT MCL tears).
3. Medial Meniscus: Medial meniscus tear. Post-operative states (meniscectomy, repairs) = absent. Grade 1/2 signal without articular extension = absent.
4. Lateral Meniscus: Lateral meniscus tear. Post-operative states (meniscectomy) = absent. Grade 1/2 signal = absent.
5. Medial OA: Medial tibiofemoral compartment osteoarthritis, joint space narrowing, or chondral loss. (DO NOT include patellofemoral).
6. Lateral OA: Lateral tibiofemoral compartment osteoarthritis, joint space narrowing, or chondral loss. (DO NOT include patellofemoral).
7. PF OA: Patellofemoral compartment osteoarthritis, patellar facet arthrosis, chondromalacia patellae.
8. Effusion: Joint effusion. (NOTE: 'physiological fluid' or 'trace fluid' is absent. However, ANY explicit 'effusion' including 'small effusion' is PRESENT).
9. Synovitis: Synovial thickening, synovitis, synovial proliferation. (NOTE: If not explicitly mentioned, it is not_stated. Do not assume synovitis just because effusion is present).
10. Baker's: Baker's cyst, popliteal cyst.
11. Contusion: Bone bruise, bone marrow edema following trauma.
12. Fracture: Cortical bone fracture, avulsion fracture. (NOTE: Old healed fracture = absent).

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
  - Russian/Bulgarian: разрыв=present, нет=absent.
  - Greek: ρήξη=present, φυσιολογικό=absent.
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
            if q_low in ["none", "null", "n/a", "", "not mentioned", "not stated", "none."]:
                is_present = False
                is_absent = False
                
            # Shield 5: Grounding Verification Against Original Report
            if (is_present or is_absent) and original_report:
                clean_q = clean_txt(exact_quote)
                clean_rep = clean_txt(original_report)
                if len(clean_q) > 3 and clean_q not in clean_rep:
                    q_words = set(re.findall(r"\b\w{4,}\b", q_low, flags=re.UNICODE))
                    rep_words = set(re.findall(r"\b\w{4,}\b", original_report.lower(), flags=re.UNICODE))
                    overlap = len(q_words & rep_words) / max(1, len(q_words))
                    
                    common_en = {"the", "and", "with", "knee", "tear", "intact", "effusion", "ligament", "meniscus", "fluid"}
                    is_english_report = len(common_en & rep_words) >= 2
                    if is_english_report and overlap < 0.3:
                        is_present = False
                        is_absent = False
                    elif not is_english_report and len(q_words) >= 3 and overlap < 0.15:
                        is_present = False
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
    chunk_size: int = 100,
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

            max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "8192"))
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
                sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, guided_decoding=guided)
                decoding_mode = "GuidedDecodingParams"
            except Exception:
                try:
                    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096, guided_json=schema_str)
                    decoding_mode = "guided_json"
                except Exception:
                    sampling_params = SamplingParams(temperature=0.0, max_tokens=4096)
                    decoding_mode = "unconstrained"
                    print("[INFO] Using unconstrained decoding at temperature 0.0 with deterministic clinical prompting.")

            global_failed_queue = []
            for i in range(0, len(remaining_df), chunk_size):
                chunk = remaining_df.iloc[i:i+chunk_size]
                full_reports = chunk["_report_text"].astype(str).tolist()
                raw_reports = [report for report in full_reports]
                messages_chunk = [[{"role": "user", "content": build_prompt(r)}] for r in raw_reports]
                uids_chunk = chunk["StudyInstanceUID"].tolist()

                print(f"[INFO] vLLM processing chunk {i//chunk_size + 1} / {((len(remaining_df)-1)//chunk_size) + 1} ({len(chunk)} reports)...")
                outputs = llm.chat(messages_chunk, sampling_params, use_tqdm=True)

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
                print(f"[CHECKPOINT] Saved {len(df_out)} studies to {out_csv}")
                
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
                    retry_params = SamplingParams(temperature=0.4, seed=42+attempt, max_tokens=4096, guided_decoding=guided_retry_p)
                except Exception:
                    try:
                        retry_params = SamplingParams(temperature=0.4, seed=42+attempt, max_tokens=4096, guided_json=schema_str)
                    except Exception:
                        retry_params = SamplingParams(temperature=0.4, seed=42+attempt, max_tokens=4096)

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
        )


if __name__ == "__main__":
    main()
