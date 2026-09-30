import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pydicom
import torch.nn.functional as F
from . import config

def read_slot(files, d, px, n_slice=config.GROUP_SIZE, out_size=config.IMG_SIZE):
    """
    Reads n_slice DICOM files from a series directory, applies a physical mm-based 
    crop, normalizes intensity, and resizes to out_size.
    """
    n = len(files)
    if n == 0:
        return None
        
    lo, hi = (int(0.2 * (n - 1)), int(0.8 * (n - 1)))
    idx = np.unique(np.linspace(lo, hi, n_slice).astype(int)) if hi > lo else np.array([n // 2])
    while len(idx) < n_slice:
        idx = np.append(idx, idx[-1])
        
    planes = []
    for i in idx[:n_slice]:
        try:
            ds = pydicom.dcmread(os.path.join(d, files[int(i)]), force=True)
            a = ds.pixel_array.astype(np.float32)
            sl = float(getattr(ds, 'RescaleSlope', 1) or 1)
            ic = float(getattr(ds, 'RescaleIntercept', 0) or 0)
            a = a * sl + ic
        except Exception:
            a = None
        planes.append(a)
        
    got = [k for k, p in enumerate(planes) if p is not None]
    if not got:
        return None
    for k, p in enumerate(planes):
        if p is None:
            planes[k] = planes[min(got, key=lambda j: abs(j - k))]
            
    shp = planes[0].shape
    vol = np.stack(planes)
    
    if px and np.isfinite(px) and (px > 0):
        want = int(round(config.CROP_MM / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = (h // 2, w // 2)
            half = want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]
            
    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-06), 0, 1)
    
    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode='bilinear', align_corners=False)
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)

def normalise_laterality(img, plane, lat):
    """Flips Right knees to appear structurally like Left knees."""
    if lat != 'R':
        return img
    if plane in ('Coronal', 'Axial'):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])


class RSNADataset(Dataset):
    """
    Standard PyTorch Dataset for RSNA Knee MRI Abnormality Detection.
    Expects a pandas DataFrame containing study metadata, targets, and NLP weights.
    """
    def __init__(self, df, data_root, is_train=True):
        self.df = df
        self.data_root = data_root
        self.is_train = is_train
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        study_id = row['StudyInstanceUID']
        laterality = row.get('laterality', 'L')
        
        # Initialize empty tensors for the 6 acquisition slots
        imgs = torch.zeros((config.N_SLOTS, config.GROUP_SIZE, config.IMG_SIZE, config.IMG_SIZE), dtype=torch.uint8)
        masks = torch.zeros(config.N_SLOTS, dtype=torch.float32)
        
        # Placeholder for actual series matching logic (maps slots to DICOM directories)
        # In a full implementation, you would iterate over config.SLOTS and read the corresponding DICOMs.
        # For each valid slot:
        # img = read_slot(files, dir_path, pixel_spacing)
        # img = normalise_laterality(img, plane, laterality)
        # imgs[slot_idx] = img
        # masks[slot_idx] = 1.0
        
        targets = torch.tensor(row[config.TARGETS].values.astype(np.float32)) if self.is_train else torch.zeros(len(config.TARGETS))
        weights = torch.tensor(row[[f"{t}_weight" for t in config.TARGETS]].values.astype(np.float32)) if self.is_train else torch.ones(len(config.TARGETS))
        
        return imgs, masks, targets, weights
