"""Central configuration.

Legacy constants (SEED, TARGETS, SLOTS, ...) are unchanged so existing code keeps working.
All preprocessing behaviour lives in PreCfg; two presets exist:

  'public' : bit-compatible re-implementation of the public 0.943 reader (for weights trained with it
             and for A/B ablations)
  'v2'     : dataset-verified upgrade (physical-mm z sampling, position ordering, stack-wide robust
             normalisation, border-median padding, area resampling, reorientation, wide-frame centring)
"""
from dataclasses import dataclass, replace

# ------------------------------------------------------------------ legacy training constants
SEED = 2026
EPOCHS = 10
BATCH_SIZE = 8
LR_HEAD = 0.001
LR_BACKBONE = 8e-06
UNFREEZE_LAST = 6
WEIGHT_DECAY = 0.02
TIME_BUDGET_HOURS = 8.0          # was referenced by train.py but never defined

AUG_ROT_DEG = 8.0
AUG_SCALE = 0.08
AUG_SHIFT = 0.05
AUG_INTENSITY = 0.1

TARGETS = [
    'ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
    'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
    'Contusion', 'Fracture',
]

# name, plane, fluid-weighted (PD/T2), fat-suppressed
SLOTS = [
    ('SAG_FLUID_FS', 'Sagittal', True, True),
    ('COR_FLUID_FS', 'Coronal', True, True),
    ('AX_FLUID_FS', 'Axial', True, True),
    ('SAG_FLUID_NOFS', 'Sagittal', True, False),
    ('COR_T1', 'Coronal', False, False),
    ('SAG_T1', 'Sagittal', False, False),
]
N_SLOTS = len(SLOTS)

# ------------------------------------------------------------------ dataset facts (verified by the audit)
# One in-plane orientation code per plane in 100% of the 24,371 train series (and the test example).
# First letter = direction image COLUMNS increase toward, second = direction image ROWS increase toward (LPS).
CANON_ORIENT = {'Sagittal': 'PI', 'Coronal': 'LI', 'Axial': 'LP'}
LAT_MIN_OFFSET_MM = 20.0         # |patient-x of image centre| needed to call laterality from geometry
COMPETITION = 'rsna-knee-abnormality-detection'
ROOT_CANDIDATES = [
    f'/kaggle/input/competitions/{COMPETITION}',
    f'/kaggle/input/{COMPETITION}',
]


@dataclass(frozen=True)
class PreCfg:
    name: str = 'v2'
    # ---- in-plane
    crop_mm: float = 130.0
    img_size: int = 336
    pad_mode: str = 'border_median'      # 'border_median' | 'public' (no pad: skip crop if window > frame)
    resize_mode: str = 'area'            # 'area' (antialiased down / linear up) | 'bilinear' (public)
    wide_center: bool = True             # centre on foreground when frame aspect > wide_ratio
    wide_ratio: float = 1.3
    # ---- intensity
    norm_lo: float = 1.0
    norm_hi: float = 99.5
    clip_negative: bool = True
    # ---- through-plane
    order_mode: str = 'position'         # 'position' (sign-normalised) | 'normal' (public: dot(ipp, r x c))
    z_mode: str = 'mm'                   # 'mm' physical grid | 'index' (public linspace over band)
    stack_depth: int = 24                # slices stored per slot
    z_step_mm: float = 4.0               # spacing between stored slices (z_mode='mm')
    band: tuple = (0.2, 0.8)             # used when z_mode='index'
    # ---- windows (2.5D)
    group: int = 3                       # adjacent slices per window (channels)
    win_stride: int = 2
    # ---- canonicalisation
    reorient: bool = True                # map to CANON_ORIENT
    lat_canon: bool = True               # mirror right knees to left
    # ---- slot logic
    slot_prefer_2d: bool = True
    slot_fs_priority: str = 'hdr'        # 'hdr' (public rule, header evidence first) | 'csv'
    slot_csv_fallback: bool = True       # use CSV flags when the header carries no usable text

    @property
    def n_windows(self) -> int:
        return max(1, (self.stack_depth - self.group) // self.win_stride + 1)


PRESETS = {
    'public': PreCfg(name='public', pad_mode='public', resize_mode='bilinear', wide_center=False,
                     norm_lo=1.0, norm_hi=99.0, clip_negative=False, order_mode='normal', z_mode='index',
                     stack_depth=12, band=(0.2, 0.8), win_stride=1, reorient=False, slot_prefer_2d=False,
                     slot_csv_fallback=False),
    'v2': PreCfg(name='v2'),
}


def get_cfg(name: str = 'v2', **overrides) -> PreCfg:
    base = PRESETS[name]
    return replace(base, **overrides) if overrides else base


# legacy aliases (kept for old imports)
CROP_MM = PRESETS['v2'].crop_mm
IMG_SIZE = PRESETS['v2'].img_size
GROUP_SIZE = PRESETS['v2'].group


def cfg_from_dict(d):
    """Rebuild a PreCfg from a JSON-loaded dict (lists -> tuples)."""
    d = dict(d)
    if 'band' in d:
        d['band'] = tuple(d['band'])
    return PreCfg(**d)
