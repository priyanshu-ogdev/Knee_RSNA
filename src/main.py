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
    if not os.path.exists(pseudo_csv):
        raise FileNotFoundError(f"[FATAL] Pseudo-labels not found at {pseudo_csv}. You MUST run nlp_extractor.py first!")
    build_labels(data_root, extra_csv=pseudo_csv, extra_weight=1.0, out_csv=final_labels_csv)
    
    # Build cache
    print("Building Index...")
    idx_dir = os.path.join(work_dir, 'idx')
    index_out = runner.get_index(data_root, idx_dir, splits=('train',), force=False)
    
    cache_dir = os.path.join(work_dir, 'cache_v2')
    os.makedirs(cache_dir, exist_ok=True)
    

    
    print("Building Memmap Cache (this takes time but drastically speeds up training)...")
    train_studies = sorted(index_out['ann'][index_out['ann']['split'] == 'train']['StudyInstanceUID'].unique())
    cfg = config.get_cfg("v2")
    cpu_cores = max(2, os.cpu_count() or 2)
    
    # SOTA Fix: The prefix must be the file stem inside the cache_dir, not the cache_dir itself!
    cache_prefix = os.path.join(cache_dir, "train")
    cache, stats = runner.run_cache(
        index_out, 'train', cache_prefix, cfg=cfg,
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
        splits.group_folds(labels_csv, folds_csv, config.TARGETS, folds=5, seed=config.SEED, group_col='site')
    
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
import datetime
def main():
    # Dynamically resolve project root relative to this script (src/scripts/run_full_pipeline.py)
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    global_start_time = time.time()
    
    # Configure logging
    import logging
    log_file = os.path.join(PROJECT_ROOT, f"pipeline_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    # Redirect print to logging for capture
    def logged_print(*args, **kwargs):
        msg = " ".join(str(a) for a in args)
        logging.info(msg)
    global print
    print = logged_print

    
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
    
    # 1. Check for Pre-computed Labels (NLP Extractor is now decoupled)
    pseudo_csv = os.path.join(DATA_ROOT, "pseudo_labels.csv")
    if not os.path.exists(pseudo_csv):
        print(f"[WARNING] Pre-computed {pseudo_csv} not found in DATA_ROOT.")
        print("[WARNING] Training will proceed with ONLY 58 Gold labels (High Risk of Overfitting!).")
        print("[INFO] Did you forget to run 'python src/data/preprocess/nlp_extractor.py' first?")
    else:
        print(f"[SUCCESS] Found pre-computed pseudo-labels at {pseudo_csv}")
    
    # 2. Preparation (Cache + Merge)
    labels_csv, cache_dir = run_preparation(DATA_ROOT, WORK_DIR, pseudo_csv)
    
    # 3. Training
    run_all_folds(labels_csv, cache_dir, WORK_DIR)
    
    total_time = time.time() - global_start_time
    t_m, t_s = divmod(int(total_time), 60)
    t_h, t_m = divmod(t_m, 60)
    
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETED SUCCESSFULLY IN {t_h:02d}h {t_m:02d}m {t_s:02d}s")
    print(f"Logs saved to {log_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
