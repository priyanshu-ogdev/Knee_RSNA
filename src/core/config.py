"""Central configuration for preprocessing, training, and memory safeguards.

Training hyperparameters are starting values and require empirical comparison
on the competition's Gold-labeled validation data.
"""
import os
from dataclasses import dataclass, replace

# ─────────────────────────────────── Hardware Safety & Circuit Breaker ────────
# Use 88 GiB as an operational target on the 128-GiB unified-memory host.
# It is advisory; the hard ceiling and available-memory guard preserve recovery room.
MEMORY_TARGET_GB = float(os.environ.get("RSNA_MEMORY_TARGET_GB", "88.0"))
CIRCUIT_BREAKER_MAX_RAM_GB = float(os.environ.get("RSNA_MAX_RAM_GB", "100.0"))
MIN_AVAILABLE_RAM_GB = float(os.environ.get("RSNA_MIN_AVAILABLE_RAM_GB", "20.0"))

# ───────────────────────────────────────────── training constants ──────────────
SEED             = 2026
EPOCHS           = 10          # Initial budget; convergence and early stopping require empirical validation
BATCH_SIZE       = 24
NUM_WORKERS      = 8
PREFETCH_FACTOR  = 1
EVAL_BATCH_SIZE  = 8
PREPROCESS_WORKERS = 8
GRAD_ACCUM       = 1           # Effective batch equals BATCH_SIZE when no final partial group exists
N_WINDOWS_TRAIN  = 5           # Number of depth-stratified windows sampled per slot and epoch
LR_HEAD          = 2e-3        # Starting value; not tuned by a DGX validation sweep
LR_BACKBONE      = 1e-5        # Starting value; scaled per-layer by the DINOv2 optimizer
WEIGHT_DECAY     = 0.05
COATNET_LR_HEAD      = 1e-3
COATNET_LR_BACKBONE  = 3e-5
COATNET_WEIGHT_DECAY = 0.02
COATNET_INPUT_SIZE   = 384
COATNET_ENCODE_CHUNK = 8
UNFREEZE_LAST    = 8           # Top 8 blocks fine-tuned with LLRD; blocks 0-3 frozen (preserves 2.5D MRI adaptation)
LORA_RANK        = 0          # Zero disables LoRA adapters
LORA_ALPHA       = 32          # Scaling used if a positive LoRA rank is selected
TIME_BUDGET_HOURS = 9999.0     # DGX Spark: no Kaggle time cap — disabled
EARLY_STOP_PATIENCE = 3        # Early stopping patience on validation Macro-AUC

# SWA duplicates model state and performs an extra full training pass. Enable
# it only for a measured ablation.
SWA_EPOCHS      = 0

# Asymmetric Loss defaults (Ridnik et al., ICCV 2021)
ASL_GAMMA_NEG   = 4.0          # Shared negative focusing exponent; compare with BCE on matched folds
ASL_GAMMA_POS   = 0.0
ASL_CLIP        = 0.05         # Negative probability shift; a hypothesis for noisy labels, not a verified gain

# Label smoothing (Szegedy et al. 2016; standard ViT recipe)
LABEL_SMOOTHING = 0.0          # Disabled in the default training recipe

# Augmentation (unchanged from baseline)
AUG_ROT_DEG  = 8.0
AUG_SCALE    = 0.08
AUG_SHIFT    = 0.05
AUG_INTENSITY = 0.1

# ─────────────────────────────────────────────── target / slot schema ────────
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]

# (name, plane, fluid-weighted, fat-suppressed)
SLOTS = [
    ("SAG_FLUID_FS",   "Sagittal",  True,  True),
    ("COR_FLUID_FS",   "Coronal",   True,  True),
    ("AX_FLUID_FS",    "Axial",     True,  True),
    ("SAG_FLUID_NOFS", "Sagittal",  True,  False),
    ("COR_T1",         "Coronal",   False, False),
    ("SAG_T1",         "Sagittal",  False, False),
]
N_SLOTS = len(SLOTS)

# ─────────────────────── dataset facts (verified by the 2026 integrity audit) ─
CANON_ORIENT       = {"Sagittal": "PI", "Coronal": "LI", "Axial": "LP"}
LAT_MIN_OFFSET_MM  = 20.0
COMPETITION        = "rsna-knee-abnormality-detection"
ROOT_CANDIDATES    = [
    f"/kaggle/input/competitions/{COMPETITION}",
    f"/kaggle/input/{COMPETITION}",
    "d:/Knee_RSNA_Data",
    "d:/Knee_RSNA/data",
    "d:/Knee_RSNA",
]


# ─────────────────────────────────────────────── PreCfg (preprocessing) ───────
@dataclass(frozen=True)
class PreCfg:
    name: str = "v2"
    # in-plane
    crop_mm: float      = 130.0
    img_size: int       = 336
    pad_mode: str       = "border_median"
    resize_mode: str    = "area"
    wide_center: bool   = True
    wide_ratio: float   = 1.3
    # intensity
    norm_lo: float      = 1.0
    norm_hi: float      = 99.5
    clip_negative: bool = True
    # through-plane
    order_mode: str     = "position"
    z_mode: str         = "mm"
    stack_depth: int    = 24
    z_step_mm: float    = 4.0
    band: tuple         = (0.2, 0.8)
    # 2.5D windows
    group: int          = 3
    win_stride: int     = 2
    # canonicalisation
    reorient: bool      = True
    lat_canon: bool     = True
    # slot logic
    slot_prefer_2d: bool    = True
    slot_fs_priority: str   = "csv"
    slot_csv_fallback: bool = True

    @property
    def n_windows(self) -> int:
        return max(1, (self.stack_depth - self.group) // self.win_stride + 1)


PRESETS = {
    "public": PreCfg(
        name="public", pad_mode="public", resize_mode="bilinear",
        wide_center=False, norm_lo=1.0, norm_hi=99.0, clip_negative=False,
        order_mode="normal", z_mode="index", stack_depth=12, band=(0.2, 0.8),
        win_stride=1, reorient=False, slot_prefer_2d=False, slot_csv_fallback=False,
    ),
    "v2": PreCfg(name="v2", img_size=518, stack_depth=32),
}


def get_cfg(name: str = "v2", **overrides) -> PreCfg:
    base = PRESETS[name]
    return replace(base, **overrides) if overrides else base


def cfg_from_dict(d: dict) -> PreCfg:
    """Rebuild a PreCfg from a JSON-loaded dict (lists → tuples)."""
    d = dict(d)
    if "band" in d:
        d["band"] = tuple(d["band"])
    return PreCfg(**d)


# Legacy aliases
CROP_MM    = PRESETS["v2"].crop_mm
IMG_SIZE   = PRESETS["v2"].img_size
GROUP_SIZE = PRESETS["v2"].group
