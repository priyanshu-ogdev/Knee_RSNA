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
