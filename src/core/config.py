"""Central configuration for preprocessing, training, and memory safeguards.

Training hyperparameters are starting values and require empirical comparison
on the competition's Gold-labeled validation data.
"""
import os
from dataclasses import dataclass, replace

# ─────────────────────────────────── Hardware Safety & Circuit Breaker ────────
# Unified memory safety constraints: total combined RAM + VRAM must stay strictly < 90 GiB.
MEMORY_TARGET_GB = float(os.environ.get("RSNA_MEMORY_TARGET_GB", "75.0"))
CIRCUIT_BREAKER_MAX_RAM_GB = float(os.environ.get("RSNA_MAX_RAM_GB", "88.0"))
MIN_AVAILABLE_RAM_GB = float(os.environ.get("RSNA_MIN_AVAILABLE_RAM_GB", "15.0"))

# ───────────────────────────────────────────── training constants ──────────────
SEED             = 2026
EPOCHS           = 12          # Full-family default; early stopping remains enabled
BATCH_SIZE       = 16          # DGX Spark optimized: utilizes 70-85 GB unified memory
NUM_WORKERS      = 8           # 8 workers balances peak throughput with IPC shared-memory stability
PREFETCH_FACTOR  = 2           # Workers queue 2 batches ahead
EVAL_BATCH_SIZE  = 8           # Controlled validation memory footprint
PREPROCESS_WORKERS = 8
GRAD_ACCUM       = 2           # Effective batch = 32 studies; rock-solid gradient estimate
N_WINDOWS_TRAIN  = 8           # 8 windows per slot: 112.6% joint anatomy coverage + perfect 3×128 chunk alignment
LR_HEAD          = 1e-3        # Lowered from 2e-3; head is randomly init'd, 2e-3 overshoots
LR_BACKBONE      = 3e-5        # Raised from 1e-5; safe with corrected LLRD direction
WEIGHT_DECAY     = 0.05
COATNET_LR_HEAD      = 1e-3
COATNET_LR_BACKBONE  = 3e-5
COATNET_WEIGHT_DECAY = 0.02
COATNET_INPUT_SIZE   = 384
COATNET_ENCODE_CHUNK = 32      # 4× larger micro-batch; fills GB10 HBM bandwidth
UNFREEZE_LAST    = 10          # Top 10/12 blocks fully unfrozen; deeper domain adaptation
LORA_RANK        = 0           # Zero disables LoRA adapters
LORA_ALPHA       = 32          # Scaling used if a positive LoRA rank is selected
TIME_BUDGET_HOURS = 9999.0     # DGX Spark: no Kaggle time cap — disabled
EARLY_STOP_PATIENCE = 5        # Raised from 3; 12 gold val studies = noisy AUC signal

# SWA averages weights from epochs (n_ep - SWA_EPOCHS) onward.
# Activates at epoch 9 (12-3). Gracefully skipped if early stop fires <=8.
# Typical gain: +0.003-0.010 macro-AUC at essentially zero extra compute cost.
SWA_EPOCHS      = 3

# Asymmetric Loss - per-class gamma_neg based on gold label neg:pos ratios (58 gold studies).
# Formula: min(3.0, max(0.0, ln(neg/pos) * 1.5)) ordered by config.TARGETS.
# Effusion 0.7:1 (more positives) -> gamma=0.0. MCL 5.4:1 (rare) -> gamma=2.5.
ASL_GAMMA_NEG   = [0.5, 2.5, 0.3, 0.6, 1.6, 2.1, 0.9, 0.0, 0.1, 2.0, 1.1, 1.2]
#                   ACL  MCL  MMen LMen MOA  LOA  PFOA Eff  Syn  Bak  Con  Frac
ASL_GAMMA_POS   = 0.0
ASL_CLIP        = 0.0          # Disabled: genuine rare negatives must not be discarded
AUC_RANK_WEIGHT  = 0.25        # Stronger direct ROC-AUC ranking gradient
AUC_RANK_MARGIN  = 0.1         # Pos must exceed neg by 0.1 logit; buffer for noisy pseudo batches

# Label smoothing: disabled for ROC-AUC ranking objectives.
# ASL focusing and PairwiseAUC require uncompressed logit margins; smoothing causes
# negative-gradient reversal on true negatives (penalizing confident predictions).
LABEL_SMOOTHING = 0.0

# Pseudo-label weight boost at training time (CSV has conservative extra_weight=0.5).
# Qwen2.5-72B multilingual medical NLP precision ~85% -> 0.5 is conservative.
# 0.5 * 1.3 = 0.65; clamped to 0.95 (below gold threshold >=0.99).
PSEUDO_WEIGHT_SCALE = 1.3

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

def resolve_data_root(provided_root: str) -> str:
    """Resolves the actual dataset root containing train.csv."""
    import os
    if os.path.exists(os.path.join(provided_root, 'train.csv')):
        return provided_root
    
    kagglehub_path = os.path.join(provided_root, 'competitions', COMPETITION)
    if os.path.exists(os.path.join(kagglehub_path, 'train.csv')):
        return kagglehub_path
        
    return provided_root

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
    "v2": PreCfg(name="v2", img_size=518, stack_depth=24),
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
