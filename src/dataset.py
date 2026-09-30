import os
import torch
import numpy as np
import pydicom
import torch.nn.functional as F
import config

def read_slot(files, d, px, n_slice=config.GROUP_SIZE, out_size=config.IMG_SIZE):
    """
    Reads 
_slice DICOM files from a series directory, applies a physical mm-based 
    crop, normalizes intensity, and resizes to out_size.
    """
    n = len(files)
    if n == 0:
        return None
        
    # Pick slice indices spread across the central band
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
        
    # Fill missing with zeros/nearest valid
    got = [k for k, p in enumerate(planes) if p is not None]
    if not got:
        return None
    for k, p in enumerate(planes):
        if p is None:
            planes[k] = planes[min(got, key=lambda j: abs(j - k))]
            
    shp = planes[0].shape
    vol = np.stack(planes)
    
    # Millimeter cropping logic
    if px and np.isfinite(px) and (px > 0):
        want = int(round(config.CROP_MM / px))
        h, w = shp
        if 16 < want < min(h, w):
            cy, cx = (h // 2, w // 2)
            half = want // 2
            vol = vol[:, max(0, cy - half):cy + half, max(0, cx - half):cx + half]
            
    # Normalize percentiles
    lo_v, hi_v = np.percentile(vol, [1, 99])
    vol = np.clip((vol - lo_v) / max(hi_v - lo_v, 1e-06), 0, 1)
    
    # Resize tensor
    t = torch.from_numpy(np.ascontiguousarray(vol)).unsqueeze(0)
    t = F.interpolate(t, size=(out_size, out_size), mode='bilinear', align_corners=False)
    return (t.squeeze(0) * 255).round().clamp(0, 255).to(torch.uint8)

def normalise_laterality(img, plane, lat):
    """
    Flips Right knees to appear structurally like Left knees.
    """
    if lat != 'R':
        return img
    if plane in ('Coronal', 'Axial'):
        return torch.flip(img, dims=[-1])
    return torch.flip(img, dims=[0])
