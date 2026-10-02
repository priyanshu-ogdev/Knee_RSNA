import os
import sys
import time
import math
import traceback
import pandas as pd
import multiprocessing as mp
from pathlib import Path

# Add the repo root to sys.path so we can import src
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src import config
from src.preprocess import runner
from src.dataset import RSNADataset
from torch.utils.data import DataLoader

def header(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)

def safe(name, fn, *a, **k):
    t0 = time.time()
    try:
        r = fn(*a, **k)
        print(f"  -> {name} done in {time.time() - t0:.1f}s")
        return r
    except Exception as e:
        traceback.print_exc()
        print(f"  [ERROR] {name}: {type(e).__name__}: {e}")
        return None

def find_data_root():
    """Find the Kaggle competition folder locally or dynamically on Kaggle."""
    try:
        return runner.discover_root()
    except FileNotFoundError:
        print("[WARNING] Could not find competition dataset via discover_root(). Using fallback paths.")
        cands = ['/kaggle/input/rsna-knee-abnormality-detection', 'd:/Knee_RSNA_Data']
        for c in cands:
            if os.path.exists(c) and (os.path.exists(os.path.join(c, 'train_series')) or os.path.exists(os.path.join(c, 'test_series'))):
                return c
        raise FileNotFoundError("train_series/ not found. Please attach the dataset.")

def step_index(DATA, out_dir):
    header("STAGE 1: SERIES INDEXING")
    cfg = config.get_cfg('v2')
    idx_dir = os.path.join(out_dir, 'idx')
    
    t0 = time.time()
    index_out = runner.get_index(DATA, idx_dir, splits=('train', 'test'), limit=0, force=True)
    total_time = time.time() - t0
    
    ann = index_out['ann']
    print(f"\n[TELEMETRY] Index built successfully in {total_time:.2f}s!")
    print(f"[TELEMETRY] Total series indexed: {len(ann)}")
    print(f"[TELEMETRY] Indexing Throughput: {len(ann) / max(total_time, 1e-6):.2f} series/sec")
    
    if 'err' in ann:
        errors = ann[ann['err'].notna()]
        if len(errors) > 0:
            print(f"\n[WARNING] {len(errors)} series had indexing errors:")
            print(errors[['SeriesInstanceUID', 'err']].head())
    return index_out

def step_cache(index_out, out_dir):
    header("STAGE 2: MEMMAP CACHE BUILD (CPU MULTICORE STRESS TEST)")
    cfg = config.get_cfg('v2')
    cache_dir = os.path.join(out_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    cpu_cores = max(2, os.cpu_count() or 2)
    print(f"Detected {cpu_cores} CPU cores. Launching workers for maximum physical throughput...")
    
    train_studies = index_out['ann'][index_out['ann']['split'] == 'train']['StudyInstanceUID'].unique()
    
    t0 = time.time()
    cache, stats = runner.run_cache(
        index_out, 
        'train', 
        cache_dir, 
        cfg=cfg, 
        workers=cpu_cores, 
        studies=train_studies, 
        fresh=True
    )
    cache_time = time.time() - t0
    
    print(f"\n[TELEMETRY] Full cache built in {cache_time:.2f}s!")
    print(f"[TELEMETRY] Total studies cached: {len(train_studies)}")
    print(f"[TELEMETRY] Cache Build Throughput: {len(train_studies) / max(cache_time, 1e-6):.2f} studies/sec")
    print(f"\n[STATS] Cache Stats:\n{stats}")
    
    cache.flush()
    return cache_dir

def step_dataloader(DATA, cache_dir):
    header("STAGE 3: PYTORCH DATALOADER THROUGHPUT (EPOCH SIMULATION)")
    cfg = config.get_cfg('v2')
    train_csv = os.path.join(DATA, 'train.csv')
    
    if not os.path.exists(train_csv):
        print("[WARNING] train.csv not found. Skipping DataLoader benchmark.")
        return
        
    df_train = pd.read_csv(train_csv)
    dataset = RSNADataset(df_train, cache_dir, cfg, is_train=True, n_windows_use=4, aug=False)
    
    batch_size = 8
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    t0 = time.time()
    total_samples = 0
    print("Starting full epoch data iteration directly from memory-mapped cache...")
    
    for batch_idx, (imgs, slot_mask, win_mask, targets, weights) in enumerate(loader):
        total_samples += imgs.shape[0]
        
        if batch_idx % 100 == 0:
            elapsed = time.time() - t0
            print(f"  Batch {batch_idx:04d} | Samples: {total_samples:05d} | Throughput: {total_samples / max(elapsed, 1e-6):.2f} samples/sec")
            
    epoch_time = time.time() - t0
    print(f"\n[TELEMETRY] DataLoader iteration completed in {epoch_time:.2f}s!")
    print(f"[TELEMETRY] Total samples batched: {total_samples}")
    print(f"[TELEMETRY] Overall Dataloader Throughput: {total_samples / max(epoch_time, 1e-6):.2f} samples/sec")

def main():
    t_start = time.time()
    
    print("=" * 100)
    print("RSNA Knee Abnormality Detection -- PIPELINE TEST BENCH (v2 Optimized)")
    print("=" * 100)
    
    DATA = safe("Locate Dataset", find_data_root)
    if not DATA:
        return
        
    print(f"\nTarget Dataset Root: {DATA}")
    OUT = os.environ.get("TEST_OUT", os.path.abspath('tmp_test_pipeline'))
    os.makedirs(OUT, exist_ok=True)
    print(f"Working Directory: {OUT}")
    
    index_out = safe("Stage 1 - Indexing", step_index, DATA, OUT)
    if not index_out:
        return
        
    cache_dir = safe("Stage 2 - Cache Build", step_cache, index_out, OUT)
    if not cache_dir:
        return
        
    safe("Stage 3 - DataLoader", step_dataloader, DATA, cache_dir)
    
    print("\n" + "=" * 100)
    print(f"ALL TESTS COMPLETED in {(time.time() - t_start) / 60:.2f} minutes")
    print("=" * 100)

if __name__ == "__main__":
    main()
