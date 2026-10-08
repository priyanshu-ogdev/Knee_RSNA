import os
import json
import re
import hashlib
import pandas as pd
import numpy as np
import torch
import time

import json

def append_to_jsonl(uid, raw_output, out_csv):
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "raw_llm_outputs.jsonl")
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"StudyInstanceUID": uid, "raw_output": raw_output}) + "\n")

import psutil
import src.core.config as config

# SOTA Fix: Load .env so the HuggingFace token (HF_TOKEN) is available for downloading the gated Nemotron model.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM, SamplingParams = None, None

# The exact targets expected by the training pipeline
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", 
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"
]

EXTRACTOR_VERSION = "clinical-report-labels-v3"


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
    """Keep only complete, in-range extraction rows for the requested studies."""
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
        valid &= values.notna() & values.between(0.0, 1.0)
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
2. MCL: Medial Collateral Ligament tear or sprain. Peri-ligamentous edema = present.
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

def clean_txt(s: str) -> str:
    # SOTA Fix: Unicode-aware stripping keeps Greek, Spanish, German, French, Dutch letters
    return re.sub(r'[^\w]', '', str(s).lower(), flags=re.UNICODE)

def parse_json_response(raw_text: str, uid: str, original_report: str = "") -> dict:
    """Safely extracts JSON from the LLM output with a Quadruple-Layer Clinical Hallucination Shield."""
    if not raw_text or not isinstance(raw_text, str):
        print(f"[ERROR] Empty raw_text for {uid}")
        return None
        
    clean_json = None
    # SOTA Fix: Case-insensitive and optional 'json' tag for markdown blocks
    md_match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw_text, re.DOTALL | re.IGNORECASE)
    if md_match:
        clean_json = md_match.group(1).strip()
    else:
        # Fallback to brace matching
        start = raw_text.find('{')
        end = raw_text.rfind('}')
        if start != -1 and end != -1 and end > start:
            clean_json = raw_text[start:end+1].strip()
            
    if not clean_json:
        print(f"[ERROR] No JSON block found in output for {uid}")
        return None
        
    try:
        # SOTA Fix: Remove trailing commas before closing braces/brackets which frequently crash json.loads
        clean_json_sanitized = re.sub(r',\s*([}\]])', r'\1', clean_json)
        try:
            data = json.loads(clean_json_sanitized)
        except json.JSONDecodeError:
            try:
                # Fallback using ast.literal_eval for non-strict python-like dicts
                import ast
                ast_str = clean_json_sanitized.replace('true', 'True').replace('false', 'False').replace('null', 'None')
                data = ast.literal_eval(ast_str)
            except Exception:
                data = None
            
        out = {"StudyInstanceUID": str(uid).strip()}
        
        # SOTA Fix: Aggressive alphanumeric key normalization to completely eliminate 
        # missing keys due to LLM hallucinating curly quotes (Baker's vs Baker's) or extra spaces.
        def normalize_key(k):
            return re.sub(r'[^a-zA-Z0-9]', '', str(k)).lower()
            
        target_map = {normalize_key(t): t for t in TARGETS}
        
        normalized_data = {}
        if data is not None:
            for k, v in data.items():
                norm_k = normalize_key(k)
                if norm_k in target_map:
                    normalized_data[target_map[norm_k]] = v
        else:
            # SOTA Fallback: Regex parsing if JSON/AST completely fails (e.g. unescaped newlines, broken quotes)
            for t in TARGETS:
                escaped_t = re.escape(t)
                pattern = rf'["\']?{escaped_t}["\']?\s*:\s*\{{([^}}]+)\}}'
                match = re.search(pattern, raw_text, re.IGNORECASE)
                if match:
                    inner = match.group(1)
                    state_match = re.search(r'["\']?state["\']?\s*:\s*["\']([^"\']+)["\']', inner, re.IGNORECASE)
                    quote_match = re.search(r'["\']?exact_quote["\']?\s*:\s*["\']([^"\']*)["\']', inner, re.IGNORECASE)
                    if state_match:
                        normalized_data[t] = {
                            "state": state_match.group(1).strip(),
                            "exact_quote": quote_match.group(1).strip() if quote_match else ""
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
                s_low = raw_state.lower().strip(' .,"\'\n')
                # Priority 1: Check for explicit not_stated / none / missing to avoid substring false positives
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
                    "positief", "presente", "vorhanden", "anwesend", "ρήξη"
                ]:
                    is_present = True
                # Priority 3: Multi-word phrase matching
                else:
                    absent_phrases = [
                        "no tear", "no fracture", "not present", "not seen", "without tear", 
                        "within normal limits", "sin rotura", "geen scheur", "keine ruptur"
                    ]
                    present_phrases = [
                        "present", "torn", "tear", "fracture", "positive", "mild", "moderate", "severe", "abnormal"
                    ]
                    if any(x in s_low for x in absent_phrases):
                        is_absent = True
                    elif any(x in s_low for x in present_phrases):
                        is_present = True
            
            # =========================================================================
            # QUADRUPLE-LAYER CLINICAL HALLUCINATION SHIELD
            # =========================================================================
            q_low = exact_quote.lower().strip()
            # ---------------------------------------------------------------------
            # Shield 1: Discard ungrounded labels with empty or "None" quote
            # ---------------------------------------------------------------------
            if q_low in ["none", "null", "n/a", "", "not mentioned", "not stated", "none."]:
                is_present = False
                is_absent = False
                
            if (is_present or is_absent) and original_report:
                # ---------------------------------------------------------------------
                  # Shield 5: Grounding Verification Against Original Report
                # ---------------------------------------------------------------------
                if (is_present or is_absent) and original_report:
                    clean_q = clean_txt(exact_quote)
                    clean_rep = clean_txt(original_report)
                    if len(clean_q) > 3 and clean_q not in clean_rep:
                        # Check word overlap if direct character substring fails
                        q_words = set(re.findall(r'\b\w{4,}\b', q_low, flags=re.UNICODE))
                        rep_words = set(re.findall(r'\b\w{4,}\b', original_report.lower(), flags=re.UNICODE))
                        overlap = len(q_words & rep_words) / max(1, len(q_words))
                        
                        # Only reject on word overlap if report is English (avoids penalizing mental translations of foreign reports)
                        common_en = {"the", "and", "with", "knee", "tear", "intact", "effusion", "ligament", "meniscus", "fluid"}
                        is_english_report = len(common_en & rep_words) >= 2
                        if is_english_report and overlap < 0.3:
                            # Fabricated quote hallucination
                            is_present = False
                            is_absent = False
                        elif not is_english_report and len(q_words) >= 3 and overlap < 0.15:
                            # Fabricated non-English quote
                            is_present = False
                            is_absent = False
            
            # Map verified findings to labels & confidence weights
            conf_str = str(v.get('confidence', '')).lower()
            if (is_present or is_absent) and original_report:
                if 'high' in conf_str:
                    out[t], out[f"{t}_weight"] = 0.95, 1.0
                elif 'low' in conf_str:
                    out[t], out[f"{t}_weight"] = 0.65, 0.5
                else:
                    out[t], out[f"{t}_weight"] = 0.85, 0.85
            elif is_absent:
                if 'high' in conf_str:
                    out[t], out[f"{t}_weight"] = 0.05, 1.0
                elif 'low' in conf_str:
                    out[t], out[f"{t}_weight"] = 0.35, 0.5
                else:
                    out[t], out[f"{t}_weight"] = 0.15, 0.85
            else:
                # not_stated / hedged / missing
                if t in ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Effusion"]:
                    out[t], out[f"{t}_weight"] = 0.0, 0.0 # Strict Mask
                else:
                    out[t], out[f"{t}_weight"] = -1.0, 0.1 # Soft Negative Marker
                    
        return out
    except Exception as e:
        print(f"[ERROR] Failed to parse JSON for {uid}: {e}")
        return None

def run_offline_extraction(data_root: str, out_csv: str, model_id: str = None):
    """Compatibility entry point using the provenance-checked, strict vLLM path."""
    return auto_complete_extraction(data_root, out_csv, model_id=model_id, engine="vllm")



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
    
    train_path = os.path.join(data_root, 'train.csv')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found at {train_path}")
        
    train_df = pd.read_csv(train_path)
    for target in TARGETS:
        if target not in train_df.columns:
            train_df[target] = pd.NA
    if train_df['StudyInstanceUID'].isna().any():
        raise ValueError("train.csv contains a missing StudyInstanceUID")
    train_df['StudyInstanceUID'] = train_df['StudyInstanceUID'].astype(str).str.strip()
    if train_df['StudyInstanceUID'].eq("").any() or train_df['StudyInstanceUID'].duplicated().any():
        raise ValueError("train.csv must have non-empty, unique StudyInstanceUID values")
    gold_mask = train_df[TARGETS].notna().any(axis=1)
    fully_labeled_mask = train_df[TARGETS].notna().all(axis=1)
    
    report_col = 'Report' if 'Report' in train_df.columns else ('report' if 'report' in train_df.columns else None)
    if report_col is None:
        raise KeyError("Could not find 'Report' or 'report' column in train.csv")
        
    reports = train_df[report_col].fillna("").astype(str)
    report_present = reports.str.strip().ne("")
    needed_df = train_df[report_present].copy() if evaluate else train_df[~fully_labeled_mask & report_present].copy()
    needed_df["_report_text"] = reports.loc[needed_df.index]
    total_needed = len(needed_df)
    needed_uids = set(needed_df['StudyInstanceUID'])
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
    
    # Resolve the engine once per run. A runtime failure must not silently switch
    # labeling methodology part-way through the dataset.
    selected_engine = engine.lower()
    requested_model = model_id if model_id else os.environ.get("LLM_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct")
    quantization = os.environ.get("NLP_QUANTIZATION", "none").lower()
    large_unquantized_model = (
        re.search(r"(?:^|[-_/])7[0-9]b(?:[-_/]|$)", requested_model.lower()) is not None
        and quantization not in {"fp8", "fp8_e4m3", "fp8_e5m2"}
    )
    if selected_engine != "vllm":
        raise ValueError("Only 'vllm' engine is supported. Rules engine has been removed for accuracy.")
    if LLM is None:
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

    # 1. Reuse only artifacts generated from this exact input/engine contract.
    existing_results = []
    done_uids = set()
    manifest_matches = False
    if os.path.exists(manifest_path) and not force:
        try:
            with open(manifest_path, encoding="utf-8") as stream:
                old_manifest = json.load(stream)
            manifest_matches = all(old_manifest.get(k) == v for k, v in contract.items())
        except Exception as e:
            print(f"[WARNING] Could not validate NLP manifest ({e}); cached labels will be regenerated.")
    if os.path.exists(out_csv) and not force and manifest_matches:
        try:
            existing_df = pd.read_csv(out_csv)
            valid_existing = _validated_extraction_rows(
                existing_df,
                needed_uids,
                report_hashes,
                {
                    "extractor_version": EXTRACTOR_VERSION,
                    "engine": selected_engine,
                    "model_id": resolved_model,
                    "prompt_sha256": prompt_sha256,
                },
            )
            done_uids = set(valid_existing['StudyInstanceUID'])
            existing_results = valid_existing.to_dict('records')
            print(
                f"[AUTO-DETECT] Valid cached extractions: {len(done_uids)} / "
                f"{total_needed}; incomplete, stale, or invalid rows will be regenerated."
            )
        except Exception as e:
            print(f"[WARNING] Could not validate cached labels ({e}). Starting fresh.")
            existing_results = []
            done_uids = set()
    elif os.path.exists(out_csv):
        print("[INFO] Cached NLP labels do not match the current input/engine contract; rebuilding them.")
            
    # 2. Check for completion
    if len(done_uids) >= total_needed:
        _atomic_csv(
            pd.DataFrame(existing_results) if existing_results else _empty_extraction_frame(),
            out_csv,
        )
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
            },
        )
        print(f"[SUCCESS] Pseudo-labels are 100% COMPLETE ({len(done_uids)} / {total_needed} studies verified).")
        print(f"[SUCCESS] File ready at: {out_csv}")
        return out_csv, {
            "status": "complete",
            "total": len(done_uids),
            "new": 0,
            "engine": selected_engine,
            "model_id": resolved_model,
        }
        
    remaining_df = needed_df[~needed_df['StudyInstanceUID'].isin(done_uids)].copy()
    print(f"[AUTO-DETECT] Remaining to extract: {len(remaining_df)} studies ({len(done_uids)/max(1, total_needed)*100:.1f}% previously done).")
    
    print(f"[CONFIG] NLP Extraction Engine Selected: '{selected_engine.upper()}'")
    print(f"[CONFIG] NLP source model/engine revision: {resolved_model}")
    
    # 4. Execution
    start_time = time.time()
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    _atomic_json(
        manifest_path,
        {**contract, "status": "in_progress", "completed_studies": len(done_uids)},
    )
        
    results = existing_results
    
    if selected_engine == "vllm":
        try:
            print(f"[INFO] Launching vLLM batch engine for {len(remaining_df)} studies...")
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

            llm = LLM(
                model=model_to_use,
                enforce_eager=enforce_eager,
                max_model_len=16384,
                tensor_parallel_size=1,
                gpu_memory_utilization=gpu_util,
            trust_remote_code=True,
                **llm_kwargs
            )
            import json
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

            
                    print("[WARNING] vLLM JSON guided decoding failed to initialize. Falling back to unconstrained decoding.")


            global_failed_queue = []
            for i in range(0, len(remaining_df), chunk_size):
                chunk = remaining_df.iloc[i:i+chunk_size]
                full_reports = chunk["_report_text"].astype(str).tolist()
                raw_reports = [report for report in full_reports]
                messages_chunk = [[{"role": "user", "content": build_prompt(r)}] for r in raw_reports]
                uids_chunk = chunk['StudyInstanceUID'].tolist()

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

                # Atomic checkpoint writes make both rows and provenance restart-safe.
                df_out = pd.DataFrame(results).drop_duplicates(subset=['StudyInstanceUID'], keep='last')
                _atomic_csv(df_out, out_csv)
                saved_rows = _validated_extraction_rows(
                    df_out,
                    needed_uids,
                    report_hashes,
                    {
                        "extractor_version": EXTRACTOR_VERSION,
                        "engine": selected_engine,
                        "model_id": resolved_model,
                        "prompt_sha256": prompt_sha256,
                    },
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
                
                # We retry with temperature=0.0 to introduce microscopic variation that can un-stick the greedy decode without hallucinating
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
                
                # Atomic save after retry pass
                df_out = pd.DataFrame(results).drop_duplicates(subset=['StudyInstanceUID'], keep='last')
                _atomic_csv(df_out, out_csv)
                saved_rows = _validated_extraction_rows(
                    df_out, needed_uids, report_hashes,
                    {"extractor_version": EXTRACTOR_VERSION, "engine": selected_engine, "model_id": resolved_model, "prompt_sha256": prompt_sha256}
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
                df_out = pd.DataFrame(results).drop_duplicates(subset=['StudyInstanceUID'], keep='last')
                _atomic_csv(df_out, out_csv)
                saved_rows = _validated_extraction_rows(
                    df_out, needed_uids, report_hashes,
                    {"extractor_version": EXTRACTOR_VERSION, "engine": selected_engine, "model_id": resolved_model, "prompt_sha256": prompt_sha256}
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
            if 'llm' in locals():
                del llm
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    final_df = (
        pd.DataFrame(results).drop_duplicates(subset=['StudyInstanceUID'], keep='last')
        if results
        else _empty_extraction_frame()
    )
    final_df = _validated_extraction_rows(
        final_df,
        needed_uids,
        report_hashes,
        {
            "extractor_version": EXTRACTOR_VERSION,
            "engine": selected_engine,
            "model_id": resolved_model,
            "prompt_sha256": prompt_sha256,
        },
    )
    missing_uids = needed_uids - set(final_df['StudyInstanceUID'])
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
    
    # Use word boundaries to prevent substring matches (e.g. "ruptur" inside "rupture")
    def has_words(words):
        return any(re.search(rf'\b{w}\b', r_lower) for w in words)
        
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

if __name__ == '__main__':
    import argparse
    import time
    from sklearn.metrics import roc_auc_score, f1_score
    
    parser = argparse.ArgumentParser(description="Run LLM NLP Label Extractor")
    parser.add_argument("--evaluate", action="store_true", help="Run in gold-evaluation mode to output ROC-AUC metrics")
    parser.add_argument("--data_root", type=str, default="", help="Path to RSNA dataset directory")
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm"], help="LLM engine to use")
    parser.add_argument("--model", type=str, default=os.environ.get("LLM_MODEL_ID", "Qwen/Qwen2.5-72B-Instruct"), help="LLM HuggingFace ID")
    parser.add_argument("--force", action="store_true", help="Force complete regeneration of all labels")
    args = parser.parse_args()

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    
    # SOTA Fix: Force Kagglehub cache to root data folder to align with main.py
    os.environ['KAGGLEHUB_CACHE'] = os.path.abspath(os.path.join(PROJECT_ROOT, 'data'))
    
    # SOTA Fix: Redirect HuggingFace cache to root data folder to avoid home directory quota limit
    if 'HF_HOME' not in os.environ:
        os.environ['HF_HOME'] = os.path.abspath(os.path.join(PROJECT_ROOT, 'data', 'hf_cache'))
    
    DATA_ROOT = args.data_root
    if DATA_ROOT and not os.path.exists(os.path.join(DATA_ROOT, 'train.csv')):
        from src.core.config import resolve_data_root
        DATA_ROOT = resolve_data_root(DATA_ROOT)

    if not DATA_ROOT or not os.path.exists(os.path.join(DATA_ROOT, 'train.csv')):
        knee_env = os.environ.get('KNEE_DATA')
        local_data = os.path.abspath(os.path.join(PROJECT_ROOT, 'data'))
        if knee_env and os.path.exists(os.path.join(knee_env, 'train.csv')):
            DATA_ROOT = os.path.abspath(knee_env)
            print(f"[SUCCESS] Dataset located via KNEE_DATA at: {DATA_ROOT}")
        elif os.path.exists(os.path.join(local_data, 'train.csv')):
            DATA_ROOT = local_data
            print(f"[SUCCESS] Dataset already present locally at: {DATA_ROOT}")
        else:
            print("Checking/Downloading RSNA dataset via Kagglehub...")
            import kagglehub
            DATA_ROOT = kagglehub.competition_download('rsna-knee-abnormality-detection')
            print(f"[SUCCESS] Dataset located at: {DATA_ROOT}")

    if args.evaluate:
        train_df = pd.read_csv(os.path.join(DATA_ROOT, 'train.csv'))
        # Evaluate on the 58 gold studies.
        gold_df = train_df[train_df['ACL'].notna()].copy()
        
        # Keep 10 studies untouched for final checks
        np.random.seed(42)
        untouched_idx = np.random.choice(gold_df.index, size=min(10, len(gold_df)), replace=False)
        eval_df = gold_df.drop(untouched_idx)
        print(f"Isolated {len(eval_df)} gold studies for evaluation. (Held out 10 for final checks).")
        
        temp_dir = "temp_gold_eval"
        os.makedirs(temp_dir, exist_ok=True)
        eval_df.to_csv(os.path.join(temp_dir, 'train.csv'), index=False)
        out_csv = os.path.join(temp_dir, 'gold_extractions.csv')
        
        print(f"Running NLP Extractor (Engine: {args.engine})...")
        import time
        start_time = time.time()
        out_path, stats = auto_complete_extraction(
            data_root=temp_dir, out_csv=out_csv, engine=args.engine,
            model_id=args.model, force=args.force, evaluate=True
        )
        elapsed = time.time() - start_time
        extracted_df = pd.read_csv(out_path)
        print(f"Extraction completed in {elapsed:.2f} seconds.")
        
        merged = pd.merge(eval_df, extracted_df, on="StudyInstanceUID", suffixes=("_true", "_pred"))
        merged['Language'] = merged['Report'].apply(detect_language)
        
        from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
        
        for lang in ['ALL'] + list(merged['Language'].unique()):
            print("\n" + "="*80)
            if lang == 'ALL':
                lang_df = merged
                print(f"OVERALL METRICS (N={len(lang_df)})")
            else:
                lang_df = merged[merged['Language'] == lang]
                print(f"METRICS FOR LANGUAGE: {lang} (N={len(lang_df)})")
            print("="*80)
            
            macro_aucs = []
            for t in TARGETS:
                y_true = lang_df[f"{t}_true"].values
                y_pred = lang_df[f"{t}_pred"].values
                # -1.0 means not_stated. We drop not_stated for precision/recall of "stated entries".
                # Also drop NaNs in true.
                valid_mask = (~np.isnan(y_true)) & (y_pred >= 0.0)
                y_true_valid = y_true[valid_mask]
                y_pred_valid = y_pred[valid_mask]
                
                coverage = np.sum(y_pred >= 0.0) / len(y_pred) if len(y_pred) > 0 else 0
                
                if len(y_true_valid) > 0:
                    preds_binary = (y_pred_valid >= 0.5).astype(int)
                    true_binary = (y_true_valid >= 0.5).astype(int)
                    p, r, f1, _ = precision_recall_fscore_support(true_binary, preds_binary, average='binary', zero_division=0)
                    
                    try:
                        auc = roc_auc_score(true_binary, y_pred_valid)
                        macro_aucs.append(auc)
                    except ValueError:
                        auc = float('nan')
                        
                    print(f"{t:18s} | Cov: {coverage*100:5.1f}% | AUC: {auc:5.3f} | P: {p:5.3f} | R: {r:5.3f} | Disagreements: {np.sum(preds_binary != true_binary)}")
                else:
                    print(f"{t:18s} | Cov: {coverage*100:5.1f}% | AUC:   N/A | P:   N/A | R:   N/A")
            
            if macro_aucs:
                print(f"\n--> MACRO AUC for {lang}: {np.nanmean(macro_aucs):.4f}")
                
    else:
        OUT = os.path.join(DATA_ROOT, "pseudo_labels.csv")
        auto_complete_extraction(
            DATA_ROOT,
            OUT,
            model_id=args.model,
            engine=args.engine,
            force=args.force
        )
