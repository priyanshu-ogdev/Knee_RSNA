"""Cache -> training/eval sample (numpy only, so it is testable without torch; dataset.py just wraps it)."""
import numpy as np

import src.core.config as config
from . import sampling


def make_sample(cache, i, cfg, train=False, n_use=None, rng=None, aug=True):
    """One study from the memmap cache.

    Returns imgs uint8 [S, W, G, H, W], slot_mask uint8 [S], win_mask bool [S, W].
    train: independent stratified-random windows per slot + one geometric/intensity augmentation per slot.
    eval : the same evenly spaced windows for every slot (n_use=None -> all windows)."""
    S, G, H = config.N_SLOTS, cfg.group, cfg.img_size
    D, stride = cfg.stack_depth, cfg.win_stride
    W = sampling.n_windows(D, G, stride) if n_use is None else min(n_use, sampling.n_windows(D, G, stride))
    imgs = np.zeros((S, W, G, H, H), np.uint8)
    wmask = np.zeros((S, W), bool)
    rng = rng if rng is not None else np.random.default_rng(0)
    for s in range(S):
        if not cache.slot[i, s]:
            continue
        starts = sampling.window_starts(D, G, stride, W, train, rng)
        win, wv = sampling.gather_windows(cache.images[i, s], cache.valid[i, s], starts, G)
        if train and aug:
            win = sampling.augment_slot(win, rng, config.AUG_ROT_DEG, config.AUG_SCALE, config.AUG_SHIFT, config.AUG_INTENSITY)
        n = len(starts)
        imgs[s, :n], wmask[s, :n] = win, wv
    slot = (np.asarray(cache.slot[i]) > 0) & wmask.any(axis=1)
    return imgs, slot.astype(np.uint8), wmask


def label_arrays(df, studies_index):
    """Targets/weights as dense float32 arrays aligned to df rows (no pandas in the hot loop).

    Missing label (NaN) -> target 0 and weight 0 (masked), fixing the NaN-loss bug (98.7% of studies are unlabeled).
    An optional '<target>_weight' column scales labelled entries (e.g. report-extractor confidence)."""
    valid_mask = df['StudyInstanceUID'].isin(studies_index)
    if not valid_mask.all():
        df = df[valid_mask].reset_index(drop=True)
    Y = np.zeros((len(df), len(config.TARGETS)), np.float32)
    Wt = np.zeros_like(Y)
    for j, t in enumerate(config.TARGETS):
        if t not in df.columns:
            continue
        y = df[t].astype(float).values
        lab = np.isfinite(y)
        Y[:, j] = np.where(lab, y, 0.0)
        w = df[f'{t}_weight'].astype(float).values if f'{t}_weight' in df.columns else np.ones(len(df))
        Wt[:, j] = np.where(lab, np.nan_to_num(w, nan=1.0), 0.0)
    rows = np.array([studies_index[s] for s in df['StudyInstanceUID']], dtype=np.int64)
    return rows, Y, Wt
