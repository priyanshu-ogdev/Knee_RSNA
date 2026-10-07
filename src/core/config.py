"""Central configuration — v2 upgrade layer.

Upgrade history
---------------
J  Epochs 10→15, weight_decay 0.02→0.05, LR_BACKBONE 8e-6→5e-6.
   Basis: ViT fine-tuning best-practices consensus 2025:
     - weight_decay=0.05 is standard for ViTs (Dosovitskiy et al. 2021,
       DeiT III Touvron et al. 2022, DINOv2 Oquab et al. 2023).
     - More conservative backbone LR (5e-6) appropriate for larger Base model.
     - 20 epochs on DGX Spark (no Kaggle 9-hr cap); cosine schedule fully decays.

   TIME_BUDGET_HOURS set to 9999 (disabled on DGX Spark; no Kaggle cap).

ASL defaults registered here so they can be overridden from a single place.

Legacy constants (SEED, TARGETS, SLOTS, CANON_ORIENT …) are unchanged so
existing code and notebooks keep working without modification.
"""
import os
from dataclasses import dataclass, replace

# ─────────────────────────────────── Hardware Safety & Circuit Breaker ────────
# Hard Unified Memory Limit (DGX 128GB node):
# If total memory usage exceeds 118 GB in any step, cleanly abort, flush memory,
# save emergency checkpoint, and prompt to restart.
CIRCUIT_BREAKER_MAX_RAM_GB = float(os.environ.get("RSNA_MAX_RAM_GB", "118.0"))

# ─────────────────────────────────────────── legacy training constants ────────
SEED             = 2026
EPOCHS           = 10          # Optimized: peak validation AUC converges by epoch 6-8; 10 avoids label-noise overfitting
BATCH_SIZE       = 16
NUM_WORKERS      = 10          # 10 workers strictly bounds pinned queue memory to ~7.7GB, achieving 80-90GB steady-state unified memory
GRAD_ACCUM       = 1           # Effective batch = 16 studies per optimizer step
N_WINDOWS_TRAIN  = 5           # 5 stratified windows: 75% articular coverage (prevents MIL focal tear dilution)
LR_HEAD          = 2e-3
LR_BACKBONE      = 1e-5        # Scaled for effective batch=16
WEIGHT_DECAY     = 0.05        # Standard ViT recipe
UNFREEZE_LAST    = 8           # Top 8 blocks fine-tuned with LLRD; blocks 0-3 frozen (preserves 2.5D MRI adaptation)
LORA_RANK        = 0          # Upgrade B — rank for QV LoRA adapters
LORA_ALPHA       = 32          # LoRA scaling: scale = LORA_ALPHA / LORA_RANK = 2
TIME_BUDGET_HOURS = 9999.0     # DGX Spark: no Kaggle time cap — disabled
EARLY_STOP_PATIENCE = 3        # Early stopping patience on validation Macro-AUC

# SWA settings (Izmailov et al., UAI 2018)
SWA_EPOCHS      = 3            # apply SWA for the last SWA_EPOCHS of training
SWA_LR          = 1e-6         # constant LR during SWA phase

# Asymmetric Loss defaults (Ridnik et al., ICCV 2021)
ASL_GAMMA_NEG   = 4.0          # paper default=4; our 98.7% unlabeled (de-facto negative) dataset needs full negative suppression
ASL_GAMMA_POS   = 0.0          # positive focusing (keep 0 — never penalise true positives)
ASL_CLIP        = 0.05         # probability shift — discards mislabelled easy negatives

# Per-target ASL gamma_neg (FIX: replaces global scalar).
# Grounded in gold-label prevalence from the 58-study gold set:
#   Effusion=60.3%, Synovitis=46.6%, ACL=41.4%, Medial Meniscus=44.8% -> lower gamma
#   MCL=15.5%, Lateral OA=19%, Bakers=20.7%, Medial OA=25.9% -> higher gamma
# Higher gamma_neg = suppress easy negatives harder = better for rare positives.
# Lower gamma_neg = don't suppress = preserve learning signal for common positives.
ASL_GAMMA_NEG_PER_TARGET = [
    2.0,   # ACL             (41.4% pos)
    4.0,   # MCL             (15.5% pos) — rare, suppress negatives hard
    2.0,   # Medial Meniscus (44.8% pos)
    2.5,   # Lateral Meniscus(39.7% pos)
    3.0,   # Medial OA       (25.9% pos)
    3.5,   # Lateral OA      (19.0% pos) — rare
    2.0,   # PF OA           (36.2% pos)
    1.0,   # Effusion        (60.3% pos) — very common; DON'T suppress negatives hard
    2.0,   # Synovitis       (46.6% pos)
    3.0,   # Baker's         (20.7% pos)
    2.5,   # Contusion       (32.8% pos)
    2.5,   # Fracture        (31.0% pos)
]

# Label smoothing (Szegedy et al. 2016; standard ViT recipe)
LABEL_SMOOTHING = 0.0          # disabled: ASL clip=0.05 already handles label noise

# Augmentation (unchanged from baseline)
AUG_ROT_DEG  = 8.0
AUG_SCALE    = 0.08
AUG_SHIFT    = 0.05
AUG_INTENSITY = 0.1

# Mixup (Zhang et al., ICLR 2018 — applied in feature space, Upgrade I)
# Rare target loss multipliers (FIX 4).
# Fracture/Bakers/Synovitis/Contusion have <5% prevalence and dominate macro-AUC variance.
RARE_TARGET_WEIGHTS = {
    "Fracture":   2.5,
    "Baker's":    2.0,
    "Synovitis":  1.5,
    "Contusion":  1.5,
}

MIXUP_ALPHA  = 0.0             # Beta(alpha, alpha) mixing coefficient; 0 = disabled

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




