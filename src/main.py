"""
RSNA 2026: End-to-End Master Training Pipeline
----------------------------------------------
Executes the entire SOTA pipeline in one shot:
1. Extracts pseudo-labels from 4,349 reports via Gemini API.
2. Merges them with the 58 gold labels into a unified training CSV.
3. Builds the optimized memory-mapped cache for ultra-fast I/O.
4. Trains a 5-fold DINOv2-based model.
"""
import os
# Force kagglehub to download directly into our root data/ directory instead of C:\Users\...
os.environ['KAGGLEHUB_CACHE'] = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

import kagglehub
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

import os
import sys
import time
import json
import traceback
import concurrent.futures
import time
import pandas as pd
import numpy as np
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

# Add project root to path (2 levels up from src/scripts/)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from google import genai
from google.genai import types

import src.core.config as config
from src.data.labels import build_labels
from src.data.preprocess import runner
from src.training.train import run_training


# ==============================================================================
# PHASE 1: NLP PSEUDO-LABEL EXTRACTION
# ==============================================================================
def build_prompt(report: str) -> str:
    return f"""You are an expert MSK radiologist. Extract the presence of 12 knee abnormalities from the following MRI radiology report. 
The report may be in any language (English, Spanish, Dutch, German, etc.). Translate mentally if needed.

TARGETS:
- ACL (Anterior Cruciate Ligament tear/injury)
- MCL (Medial Collateral Ligament tear/injury)
- Medial Meniscus (tear/injury)
- Lateral Meniscus (tear/injury)
- Medial OA (Medial compartment Osteoarthritis / cartilage loss)
- Lateral OA (Lateral compartment Osteoarthritis / cartilage loss)
- PF OA (Patellofemoral Osteoarthritis / cartilage loss)
- Effusion (Joint fluid)
- Synovitis (Synovial thickening/inflammation)
- Baker's (Baker's cyst / Popliteal cyst)
- Contusion (Bone bruise/contusion)
- Fracture (Bone fracture)

RULES:
1. Output MUST be valid JSON matching the exact output schema.
2. "state": EXACTLY ONE of ["present", "absent", "not_stated"].
3. "present" = explicitly torn/injured. "absent" = explicitly normal. "not_stated" = omitted.
4. Watch out for negations (e.g. "no evidence of ACL tear" -> ACL=absent).

REPORT:
{report}

OUTPUT SCHEMA:
{{
  "ACL": {{"state": "..."}},
  "MCL": {{"state": "..."}},
  "Medial Meniscus": {{"state": "..."}},
  "Lateral Meniscus": {{"state": "..."}},
  "Medial OA": {{"state": "..."}},
  "Lateral OA": {{"state": "..."}},
  "PF OA": {{"state": "..."}},
  "Effusion": {{"state": "..."}},
  "Synovitis": {{"state": "..."}},
  "Baker's": {{"state": "..."}},
  "Contusion": {{"state": "..."}},
  "Fracture": {{"state": "..."}}
}}
"""

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(Exception)
)
def _call_gemini_with_retry(client, report: str):
    return client.models.generate_content(
        model='gemini-3.1-pro',
        contents=build_prompt(report),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
        ),
    )

def extract_study(client, row):
    try:
        response = _call_gemini_with_retry(client, row['Report'])
        
        # Strip potential markdown formatting (e.g. ```json ... ```)
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        data = json.loads(raw_text.strip())
        
        # Format for labels.py (value and confidence weight)
        out = {"StudyInstanceUID": row["StudyInstanceUID"]}
        for t in config.TARGETS:
            if t in data and "state" in data[t]:
                state = data[t]["state"]
                
                if state == "present":
                    out[t], out[f"{t}_weight"] = 1.0, 0.5
                elif state == "absent":
                    out[t], out[f"{t}_weight"] = 0.0, 0.5
                else: # not_stated
                    if t in ["ACL", "Medial Meniscus", "Lateral Meniscus", "Effusion", "MCL"]:
                        out[t], out[f"{t}_weight"] = 0.0, 0.0 # MASK
                    else:
                        out[t], out[f"{t}_weight"] = 0.0, 0.1 # Soft negative
            else:
                out[t], out[f"{t}_weight"] = 0.0, 0.0 # MASK fallback
        return out
    except Exception as e:
        print(f"Error on {row['StudyInstanceUID']}: {e}")
        return None

def run_nlp_extraction(data_root: str, out_csv: str):
    print("=" * 80)
    print("PHASE 1: NLP PSEUDO-LABEL EXTRACTION")
    print("=" * 80)
    
    if os.path.exists(out_csv):
        print(f"[INFO] {out_csv} already exists. Skipping API extraction.")
        return out_csv
        
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required for Phase 1.")
        
    client = genai.Client(api_key=api_key)
    train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
    
    # We only need to extract reports that DON'T have gold labels.
    # If a study has ANY gold label, we can skip extracting its report to save API calls,
    # as labels.py will overwrite the pseudo-label with gold anyway.
    gold_mask = train_df[config.TARGETS].notna().any(axis=1)
    to_extract = train_df[~gold_mask & train_df['Report'].notna()].copy()
    
    print(f"Total reports to process: {len(to_extract)}")
    
    results = []
    # Gemini API free tier allows 15 RPM. Pro account limits vary but pacing is required.
    # We use 10 workers + exponential backoff + pacing to safely process 4,300+ requests.
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(extract_study, client, row): i for i, row in to_extract.iterrows()}
        for i, f in enumerate(concurrent.futures.as_completed(futures)):
            res = f.result()
            if res:
                results.append(res)
            
            # Pace the main loop slightly to avoid bursting too hard
            time.sleep(0.1)
                
            if (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(to_extract)} reports...")
                pd.DataFrame(results).to_csv(out_csv, index=False) # Checkpoint
                
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"[SUCCESS] Extraction complete. Saved to {out_csv}")
    return out_csv


# ==============================================================================
# PHASE 2: CACHE & PIPELINE PREPARATION
# ==============================================================================
def run_preparation(data_root: str, work_dir: str, pseudo_csv: str):
    print("\n" + "=" * 80)
    print("PHASE 2: DATASET MERGE & CACHE BUILD")
    print("=" * 80)
    
    # Merge gold + pseudo labels
    final_labels_csv = os.path.join(work_dir, "train_labels_v2.csv")
    print("Merging gold and pseudo labels...")
    build_labels(data_root, extra_csv=pseudo_csv, extra_weight=1.0, out_csv=final_labels_csv)
    
    # Build cache
    print("Building Index...")
    idx_dir = os.path.join(work_dir, 'idx')
    index_out = runner.get_index(data_root, idx_dir, splits=('train',), force=False)
    
    cache_dir = os.path.join(work_dir, 'cache_v2')
    if os.path.exists(cache_dir) and len(os.listdir(cache_dir)) > 0:
        print(f"[INFO] Cache already exists at {cache_dir}. Skipping cache build.")
        return final_labels_csv, cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    
    print("Building Memmap Cache (this takes time but drastically speeds up training)...")
    train_studies = sorted(index_out['ann'][index_out['ann']['split'] == 'train']['StudyInstanceUID'].unique())
    cfg = config.get_cfg("v2")
    cpu_cores = max(2, os.cpu_count() or 2)
    
    cache, stats = runner.run_cache(
        index_out, 'train', cache_dir, cfg=cfg,
        workers=cpu_cores, studies=train_studies, fresh=False
    )
    print(f"[SUCCESS] Cache ready. Stats: {stats}")
    
    return final_labels_csv, cache_dir

# ==============================================================================
# PHASE 3: 5-FOLD MODEL TRAINING
# ==============================================================================
def run_all_folds(labels_csv: str, cache_dir: str, work_dir: str):
    print("\n" + "=" * 80)
    print("PHASE 3: 5-FOLD MODEL TRAINING")
    print("=" * 80)
    
    # First generate splits
    from src.data.preprocess import splits
    folds_csv = os.path.join(work_dir, "folds.csv")
    if not os.path.exists(folds_csv):
        print("Generating 5-fold stratification splits...")
        splits.make_folds(labels_csv, folds_csv, config.TARGETS, folds=5, seed=config.SEED)
    
    cache_prefix = os.path.join(cache_dir, "train")
    
    for fold in range(5):
        print(f"\n--- STARTING FOLD {fold} ---")
        try:
            best_auc = run_training(
                labels_csv=labels_csv,
                cache_prefix=cache_prefix,
                folds_csv=folds_csv,
                fold=fold,
                out_dir=os.path.join(work_dir, f"models_fold{fold}"),
            )
            print(f"Fold {fold} finished with Best Macro-AUC: {best_auc:.4f}")
        except Exception as e:
            print(f"Error in Fold {fold}: {e}")
            traceback.print_exc()

# ==============================================================================
# MAIN ENTRY
# ==============================================================================
def main():
    # Dynamically resolve project root relative to this script (src/scripts/run_full_pipeline.py)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # DGX / Linux compatible relative paths
    print("=" * 80)
    print("PHASE 0: DATASET DOWNLOAD")
    print("=" * 80)
    print("Checking/Downloading RSNA dataset via Kagglehub...")
    # Kagglehub automatically skips downloading if the dataset is already cached locally.
    DATA_ROOT = kagglehub.competition_download('rsna-knee-abnormality-detection')
    print(f"[SUCCESS] Dataset located at: {DATA_ROOT}")
    WORK_DIR = os.environ.get('RSNA_OUT_DIR', os.path.join(PROJECT_ROOT, 'pipeline_out'))
    
    os.makedirs(WORK_DIR, exist_ok=True)
    
    print("Pipeline Output Directory:", WORK_DIR)
    
    pseudo_csv = os.path.join(WORK_DIR, "pseudo_labels.csv")
    
    # 1. NLP Extraction
    run_nlp_extraction(DATA_ROOT, pseudo_csv)
    
    # 2. Preparation (Cache + Merge)
    labels_csv, cache_dir = run_preparation(DATA_ROOT, WORK_DIR, pseudo_csv)
    
    # 3. Training
    run_all_folds(labels_csv, cache_dir, WORK_DIR)
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
