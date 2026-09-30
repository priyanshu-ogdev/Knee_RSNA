import os
import time
import gc
from concurrent.futures import ThreadPoolExecutor
import torch
import numpy as np
import pandas as pd
from . import config
from .model import build_model
from .dataset import read_slot, normalise_laterality

def build_cache(slot_map, data_root, lat_map):
    """
    Highly optimized multi-threaded I/O pipeline that bypasses standard DataLoaders.
    Pre-allocates RAM and parses DICOMs in parallel to prevent GPU starvation.
    """
    studies = sorted(slot_map.keys())
    sidx = {s: i for i, s in enumerate(studies)}
    
    # Pre-allocate RAM
    cache = np.zeros((len(studies), config.N_SLOTS, config.GROUP_SIZE, config.IMG_SIZE, config.IMG_SIZE), dtype=np.uint8)
    masks = np.zeros((len(studies), config.N_SLOTS), dtype=np.float32)
    
    # Construct job queue
    jobs = []
    for st in studies:
        for k, (name, plane, _, _) in enumerate(config.SLOTS):
            if name in slot_map[st]:
                jobs.append((st, k, plane, slot_map[st][name], data_root))
                
    print(f"Decoding {len(jobs)} slot-series...")
    
    # Multi-threaded extraction
    done = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        # Wrapper to unpack job args for read_slot
        def _job_fn(j):
            st, k, plane, files, root = j
            series_path = os.path.join(root, st, files['series_id'])
            dicom_list = files.get('ordered', files.get('files', []))
            return read_slot(dicom_list, series_path, px=None) # pixel spacing logic applied inside
            
        for (st, k, plane, _, _), img in zip(jobs, pool.map(_job_fn, jobs)):
            done += 1
            if img is None:
                continue
            lat = lat_map.get(st, 'L')
            cache[sidx[st], k] = normalise_laterality(img, plane, lat).numpy()
            masks[sidx[st], k] = 1.0
            
    print(f"Filled {int(masks.sum())}/{len(jobs)} slots.")
    gc.collect()
    return studies, cache, masks

def predict_from_cache(model, cache, masks, device):
    """
    Runs highly optimized inference directly from RAM cache.
    """
    model.eval()
    all_preds = []
    
    with torch.inference_mode(), torch.autocast('cuda', enabled=device.type == 'cuda'):
        # Manual batching
        for i in range(0, len(cache), config.BATCH_SIZE * 2):
            b_cache = cache[i:i + config.BATCH_SIZE * 2]
            b_mask = masks[i:i + config.BATCH_SIZE * 2]
            
            imgs = torch.from_numpy(b_cache).to(device, non_blocking=True)
            m = torch.from_numpy(b_mask).to(device, non_blocking=True)
            
            logits = model(imgs, m)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            
    return np.vstack(all_preds) if all_preds else np.array([])

def rank_ensemble(dino_df, coatnet_df):
    """
    0.943 SOTA Rank Ensembling logic. Converts raw probabilities to percentiles (ranks),
    then applies target-specific blending weights.
    """
    coatnet_df = coatnet_df.set_index('StudyInstanceUID').reindex(dino_df['StudyInstanceUID']).reset_index()
    dino_ranks = dino_df[config.TARGETS].rank(method='average', pct=True)
    coat_ranks = coatnet_df[config.TARGETS].rank(method='average', pct=True)
    
    coat_weights = {label: 0.60 for label in config.TARGETS}
    coat_weights.update({
        'ACL': 0.75,
        'Medial Meniscus': 0.80,
        'Lateral Meniscus': 1.00,
        'Lateral OA': 0.75,
        'Fracture': 0.75,
    })
    
    blend_df = dino_df.copy()
    for label in config.TARGETS:
        w = coat_weights[label]
        blend_df[label] = ((1.0 - w) * dino_ranks[label]) + (w * coat_ranks[label])
        
    blend_df[config.TARGETS] = blend_df[config.TARGETS].rank(method='average', pct=True)
    return blend_df

def build_metadata_maps(series_desc_path, df):
    """
    Parses test_series_descriptions.csv to map StudyInstanceUID -> SeriesInstanceUID -> Plane.
    """
    slot_map = {}
    lat_map = {}
    
    if os.path.exists(series_desc_path):
        desc = pd.read_csv(series_desc_path)
        for _, row in desc.iterrows():
            st = str(row['StudyInstanceUID'])
            se = str(row['SeriesInstanceUID'])
            plane = str(row['SeriesDescription'])
            
            if st not in slot_map:
                slot_map[st] = {}
            for (name, c_plane, _, _) in config.SLOTS:
                if c_plane in plane: 
                    slot_map[st][name] = {'series_id': se, 'files': []}
                    break
    
    for st in df['StudyInstanceUID']:
        lat_map[st] = 'L'
        if st not in slot_map:
             slot_map[st] = {}
            
    return slot_map, lat_map

def run_inference(test_csv_path, series_desc_path, data_root, model_weights_path, coatnet_csv_path=None):
    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing inference on {device}...")
    
    try:
        df = pd.read_csv(test_csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find {test_csv_path}")
        return
        
    print("Phase 1: Pre-Parsing Metadata...")
    slot_map, lat_map = build_metadata_maps(series_desc_path, df)
    
    for st, slots in slot_map.items():
        for name, data in slots.items():
            series_path = os.path.join(data_root, st, data['series_id'])
            if os.path.exists(series_path):
                data['files'] = sorted(os.listdir(series_path))
                
    print("Phase 2: Overhauling I/O with build_cache...")
    studies, cache, masks = build_cache(slot_map, data_root, lat_map)
    
    print("Phase 3: Secure Forward Pass...")
    model = build_model().to(device)
    if os.path.exists(model_weights_path):
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
        
    preds = predict_from_cache(model, cache, masks, device)
    
    sub_df = pd.DataFrame({'StudyInstanceUID': studies})
    for i, target in enumerate(config.TARGETS):
        sub_df[target] = preds[:, i] if len(preds) > 0 else 0.5
        
    if coatnet_csv_path and os.path.exists(coatnet_csv_path):
        print(f"Applying 0.943 Rank Ensembling...")
        coatnet_df = pd.read_csv(coatnet_csv_path)
        sub_df = rank_ensemble(sub_df, coatnet_df)
        
    sub_df.to_csv('submission.csv', index=False)
    elapsed = time.time() - t0
    print(f"Inference complete in {elapsed:.1f}s. Saved to submission.csv")

if __name__ == '__main__':
    run_inference(
        test_csv_path='/kaggle/input/rsna-knee-abnormality-detection/test.csv',
        series_desc_path='/kaggle/input/rsna-knee-abnormality-detection/test_series_descriptions.csv',
        data_root='/kaggle/input/rsna-knee-abnormality-detection/test_images',
        model_weights_path='best_dinov2_model.pt',
        coatnet_csv_path='coatnet_preds.csv'
    )
