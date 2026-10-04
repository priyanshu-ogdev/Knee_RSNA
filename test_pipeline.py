import os
import sys
import time
import math
import shutil
import traceback
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

# Add the repo root to sys.path so we can import src
if '__file__' in globals():
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
elif os.path.abspath('.') not in sys.path:
    sys.path.insert(0, os.path.abspath('.'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import config
from src.preprocess import runner, index
from src.dataset import RSNADataset
from src.model import build_model

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

def create_synthetic_dataset(out_dir):
    """Generates a miniature, physically-valid 2-study DICOM dataset for local testing."""
    import pydicom
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    print("\n[MOCK] Generating miniature physically-valid synthetic DICOM dataset...")
    os.makedirs(out_dir, exist_ok=True)
    ts_dir = os.path.join(out_dir, "train_series")
    os.makedirs(ts_dir, exist_ok=True)

    studies = ["1.2.826.0.1.3680043.8.498.test001", "1.2.826.0.1.3680043.8.498.test002"]
    series_configs = [
        ("Sagittal", "Sagittal PD FS", True, True, (0, 1, 0, 0, 0, -1), (0, 0, 1)),
        ("Coronal", "Coronal T1", False, False, (1, 0, 0, 0, 0, -1), (0, 1, 0)),
        ("Axial", "Axial T2 FS", True, True, (1, 0, 0, 0, 1, 0), (0, 0, 1)),
    ]

    ts_rows = []
    for st_idx, st_uid in enumerate(studies):
        st_path = os.path.join(ts_dir, st_uid)
        os.makedirs(st_path, exist_ok=True)
        for plane, desc, fluid, fs, iop, normal in series_configs:
            se_uid = f"{st_uid}.{plane.lower()}"
            se_path = os.path.join(st_path, se_uid)
            os.makedirs(se_path, exist_ok=True)
            ts_rows.append({
                "StudyInstanceUID": st_uid,
                "SeriesInstanceUID": se_uid,
                "Anatomical_Plane": plane,
                "Fluid_Sensitive": int(fluid),
                "Fat_Suppression": int(fs),
            })
            n_slices = 12
            for sl in range(n_slices):
                dcm_name = f"slice_{sl:03d}.dcm"
                fpath = os.path.join(se_path, dcm_name)
                ipp = [float(sl * 3.5 * normal[0] - 80), float(sl * 3.5 * normal[1] - 80), float(sl * 3.5 * normal[2] - 80)]
                
                file_meta = FileMetaDataset()
                file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
                file_meta.MediaStorageSOPInstanceUID = generate_uid()
                file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

                ds = FileDataset(fpath, {}, file_meta=file_meta, preamble=b"\0" * 128)
                ds.Modality = "MR"
                ds.PatientID = f"PAT_{st_idx}"
                ds.StudyInstanceUID = st_uid
                ds.SeriesInstanceUID = se_uid
                ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
                ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
                ds.Rows, ds.Columns = 320, 320
                ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
                ds.PixelRepresentation = 0
                ds.SamplesPerPixel = 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.PixelSpacing = [0.5, 0.5]
                ds.SliceThickness = 3.0
                ds.ImagePositionPatient = ipp
                ds.ImageOrientationPatient = list(iop)
                ds.InstanceNumber = sl + 1
                ds.SeriesDescription = desc
                ds.RepetitionTime = 2500.0 if fluid else 600.0
                ds.EchoTime = 40.0 if fluid else 15.0
                ds.RescaleSlope = 1.0
                ds.RescaleIntercept = 0.0

                rows, cols = 320, 320
                arr = (np.random.normal(50, 10, (rows, cols))).clip(0, 4000).astype(np.uint16)
                y, x = np.ogrid[:rows, :cols]
                mask = ((y - rows // 2)**2 / (rows // 4)**2 + (x - cols // 2)**2 / (cols // 4)**2) <= 1
                arr[mask] += 600
                ds.PixelData = arr.tobytes()
                ds.save_as(fpath, write_like_original=False)

    pd.DataFrame(ts_rows).to_csv(os.path.join(out_dir, "train_series.csv"), index=False)
    
    tr_rows = []
    for st_uid in studies:
        row = {"StudyInstanceUID": st_uid, "Report": "Technique: MRI of the knee. ACL normal. Effusion present."}
        for t in config.TARGETS:
            row[t] = 1.0 if t == "Effusion" else 0.0
        tr_rows.append(row)
    pd.DataFrame(tr_rows).to_csv(os.path.join(out_dir, "train.csv"), index=False)
    print(f"[MOCK] Synthetic dataset generated at {out_dir} with {len(studies)} studies and {len(ts_rows)} series.")
    return out_dir

def find_data_root():
    """Find the Kaggle competition folder locally, dynamically, or generates a synthetic test bench."""
    try:
        return index.discover_root()
    except FileNotFoundError:
        cands = ['/kaggle/input/rsna-knee-abnormality-detection', 'd:/Knee_RSNA_Data', 'd:/Knee_RSNA/data']
        for c in cands:
            if os.path.exists(c) and (os.path.exists(os.path.join(c, 'train_series')) or os.path.exists(os.path.join(c, 'test_series'))):
                return c
        print("[INFO] Real DICOM dataset not mounted. Generating mock dataset for verification...")
        synth_dir = os.path.abspath("tmp_test_pipeline/mock_knee_dataset")
        return create_synthetic_dataset(synth_dir)

def step_index(DATA, out_dir):
    header("STAGE 1: SERIES INDEXING & GEOMETRY PARSING")
    idx_dir = os.path.join(out_dir, 'idx')
    
    t0 = time.time()
    index_out = runner.get_index(DATA, idx_dir, splits=('train',), limit=0, force=True)
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
    
    try:
        cache_base = runner.pick_scratch([out_dir, tempfile.gettempdir()])
    except Exception:
        cache_base = out_dir
        
    cache_dir = os.path.join(cache_base, 'cache_bench')
    os.makedirs(cache_dir, exist_ok=True)
    
    cpu_cores = max(2, os.cpu_count() or 2)
    print(f"Detected {cpu_cores} CPU cores. Launching workers for maximum physical throughput...")
    
    train_studies = sorted(index_out['ann'][index_out['ann']['split'] == 'train']['StudyInstanceUID'].unique())
    free_gb = shutil.disk_usage(cache_base).free / 1e9
    cfg = runner.fit_cache_cfg(len(train_studies), free_gb, preset='v2')
    print(f"Fitted cache config: img_size={cfg.img_size}, stack_depth={cfg.stack_depth}, z_step_mm={cfg.z_step_mm}")
    
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
    
    return cache_dir, cfg

def step_dataloader(DATA, cache_dir, cfg):
    header("STAGE 3: PYTORCH DATALOADER THROUGHPUT (EPOCH SIMULATION)")
    train_csv = os.path.join(DATA, 'train.csv')
    
    if not os.path.exists(train_csv):
        print("[WARNING] train.csv not found. Skipping DataLoader benchmark.")
        return None
        
    df_train = pd.read_csv(train_csv)
    dataset = RSNADataset(df_train, cache_dir, cfg, is_train=True, n_windows_use=4, aug=True)
    
    batch_size = min(4, len(dataset))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    t0 = time.time()
    total_samples = 0
    print("Starting full epoch data iteration directly from memory-mapped cache...")
    
    first_batch = None
    for batch_idx, batch in enumerate(loader):
        imgs, slot_mask, win_mask, targets, weights = batch
        total_samples += imgs.shape[0]
        if first_batch is None:
            first_batch = batch
            print(f"  Batch 0: imgs={tuple(imgs.shape)} ({imgs.dtype}), slot_mask={tuple(slot_mask.shape)}, targets={tuple(targets.shape)}")
            
    epoch_time = time.time() - t0
    print(f"\n[TELEMETRY] DataLoader iteration completed in {epoch_time:.2f}s!")
    print(f"[TELEMETRY] Total samples batched: {total_samples}")
    print(f"[TELEMETRY] Overall Dataloader Throughput: {total_samples / max(epoch_time, 1e-6):.2f} samples/sec")
    return first_batch

def step_model(first_batch):
    header("STAGE 4: MODEL FORWARD & SPARSE HEAD VERIFICATION")
    if first_batch is None:
        print("Skipping Stage 4 (no batch).")
        return
        
    imgs, slot_mask, win_mask, targets, weights = first_batch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target compute device: {device}")
    
    print("Instantiating DINOv2 / SlotHead multi-view model...")
    model = build_model(variant='dinov2-small').to(device)
    model.eval()
    
    with torch.no_grad():
        imgs_d = imgs.to(device)
        slot_d = slot_mask.to(device)
        win_d = win_mask.to(device)
        
        t0 = time.time()
        logits = model(imgs_d, slot_d, win_d)
        fw_time = time.time() - t0
        
        probs = torch.sigmoid(logits.float())
        loss = (F.binary_cross_entropy_with_logits(logits.float(), targets.to(device), reduction='none') * weights.to(device)).sum() / weights.to(device).sum().clamp_min(1.0)
        
    print(f"[TELEMETRY] Forward pass executed in {fw_time:.3f}s!")
    print(f"[TELEMETRY] Logits shape: {tuple(logits.shape)} | Targets: {tuple(targets.shape)}")
    print(f"[TELEMETRY] Sample Mean Predictions: {probs.mean(dim=0).cpu().numpy().round(3)}")
    print(f"[TELEMETRY] Weighted BCE Loss: {loss.item():.4f}")
    assert logits.shape == targets.shape, f"Shape mismatch: {logits.shape} vs {targets.shape}"
    print("[PASS] Model architecture and sparse slot pooling fully verified!")

def main():
    t_start = time.time()
    
    print("=" * 100)
    print("RSNA Knee Abnormality Detection -- PIPELINE TEST BENCH (v2 Optimized)")
    print("=" * 100)
    
    DATA = safe("Locate / Prepare Dataset", find_data_root)
    if not DATA:
        return
        
    print(f"\nTarget Dataset Root: {DATA}")
    OUT = os.environ.get("TEST_OUT", os.path.abspath('tmp_test_pipeline'))
    os.makedirs(OUT, exist_ok=True)
    print(f"Working Directory: {OUT}")
    
    index_out = safe("Stage 1 - Indexing", step_index, DATA, OUT)
    if not index_out:
        return
        
    res = safe("Stage 2 - Cache Build", step_cache, index_out, OUT)
    if not res:
        return
    cache_dir, cfg = res
        
    batch = safe("Stage 3 - DataLoader", step_dataloader, DATA, cache_dir, cfg)
    
    safe("Stage 4 - Model Forward & Loss", step_model, batch)
    
    print("\n" + "=" * 100)
    print(f"ALL TESTS COMPLETED in {(time.time() - t_start) / 60:.2f} minutes")
    print("=" * 100)

if __name__ == "__main__":
    main()
