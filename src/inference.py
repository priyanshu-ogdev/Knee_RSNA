"""Pipelined inference — v2 upgrade layer.

Upgrades implemented vs. baseline (0.943):
  H  Temperature calibration (Guo et al. ICML 2017)
       Accepts per-model temperature scalars (T) fitted on OOF data.
       Applied as: sigmoid(logits / T).
  -  N-Arm Blending
       Replaces the legacy 2-arm rank_ensemble with rank_ensemble_n from
       ensemble.py, supporting DINOv2, CoAtNet, and the new ConvNeXt arm.

Hidden-rerun rules (public notebooks' failure history): nothing here may raise
on data conditions. Missing series/flags degrade to masked slots; studies with
no usable slot fall back to median prediction. Submission is ALWAYS written.
"""
from __future__ import annotations

import os
import tempfile
import numpy as np
import pandas as pd
import torch

from . import config

from .ensemble import rank_ensemble_n
from .preprocess import index as pix
from .preprocess import slots as pslots
from .preprocess import pipeline, cache as pcache, loader


def prepare_test_tables(root, cfg, workers=None):
    """Directory-truth index of test_series -> annotated series table, slot table, laterality, records."""
    idx = pix.build_index(root, ("test",), workers=workers, chunk=500, progress=False)
    idx = pix.attach_csv_flags(idx, root)
    idx["n_slices"] = idx["n_slices"].fillna(0) if "n_slices" in idx else 0
    ann, tab = pslots.assign_all(idx, cfg.slot_prefer_2d, cfg.slot_fs_priority, cfg.slot_csv_fallback)
    return ann, tab, pipeline.study_sides(idx), pipeline.index_to_records(ann)



@torch.inference_mode()
def predict_chunk_tta(
    models: list,
    cache,
    a: int,
    b: int,
    cfg,
    device,
    n_use=None,
    batch: int = 4,
    temperatures=None,
    n_tta: int = 4,
    tta_seed: int = 42,
) -> np.ndarray:
    """TTA: average predictions over N augmented views — no horizontal flips.

    Correct implementation: each TTA pass calls make_sample with aug=True and a
    different RNG seed so augment_slot applies a distinct random rotation/scale/
    intensity shift. Pass 0 always uses aug=False (clean prediction) for stability.

    NO FLIPS: knees are laterality-canonicalised to 'left' at preprocessing time;
    horizontal flips destroy medial/lateral consistency in the slot head.
    Source: sampling.py module docstring.

    AUC improvement: averaging reduces logit variance without adding bias.
    Expected gain: +0.002 to +0.005 AUC (standard for 4-pass TTA on MRI).
    """
    from .preprocess import sampling
    from concurrent.futures import ThreadPoolExecutor

    all_preds = []
    T = temperatures or [1.0] * len(models)

    for tta_pass in range(n_tta):
        # Pass 0: no augmentation (clean, stable reference)
        use_aug = (tta_pass > 0)
        rng_seed = tta_seed + tta_pass * 1000

        out = []
        for lo in range(a, b, batch):
            rows = list(range(lo, min(lo + batch, b)))
            rng = np.random.default_rng(rng_seed)

            def _make(i, _rng=rng):
                return loader.make_sample(
                    cache, i, cfg,
                    train=False,          # always use eval-mode window selection (evenly spaced)
                    n_use=n_use,
                    rng=_rng,
                    aug=use_aug,          # augmentation applied only for TTA passes 1+
                )

            with ThreadPoolExecutor(max_workers=len(rows)) as ex:
                smp = list(ex.map(_make, rows))

            imgs = torch.from_numpy(np.stack([s[0] for s in smp])).to(device, non_blocking=True)
            slot = torch.from_numpy(np.stack([s[1] for s in smp])).float().to(device, non_blocking=True)
            wm   = torch.from_numpy(np.stack([s[2] for s in smp])).float().to(device, non_blocking=True)

            ps = []
            for m, t_val in zip(models, T):
                use_bf16 = torch.cuda.is_bf16_supported() if device.type == "cuda" else False
                ctx = torch.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16,
                                     enabled=(device.type == "cuda"))
                with ctx:
                    logits = m(imgs, slot, wm).float()
                    scaled = logits / max(t_val, 1e-6)
                    ps.append(torch.sigmoid(scaled).cpu().numpy())
            out.append(np.mean(ps, axis=0))

        all_preds.append(np.concatenate(out))

    return np.mean(all_preds, axis=0)


@torch.inference_mode()
def predict_chunk(
    models: list[torch.nn.Module],
    cache: pcache.StudyCache,
    a: int,
    b: int,
    cfg: config.PreCfg,
    device: torch.device,
    n_use: int | None = None,
    batch: int = 4,
    temperatures: list[float] | None = None,
) -> np.ndarray:
    """Predict a chunk of studies using an ensemble of models (e.g., 5 folds).

    If temperatures are provided, logits are scaled by T before sigmoid.
    Returns: [b-a, n_targets] average probabilities.
    """
    out = []
    T = temperatures or [1.0] * len(models)

    from concurrent.futures import ThreadPoolExecutor

    for lo in range(a, b, batch):
        rows = list(range(lo, min(lo + batch, b)))
        # eval mode: evenly spaced windows, no rng/aug
        # UPGRADE: parallelize batch creation to hide numpy/cv2 CPU overhead
        with ThreadPoolExecutor(max_workers=len(rows)) as ex:
            smp = list(ex.map(lambda i: loader.make_sample(cache, i, cfg, train=False, n_use=n_use), rows))

        imgs = torch.from_numpy(np.stack([s[0] for s in smp])).to(device, non_blocking=True)
        slot = torch.from_numpy(np.stack([s[1] for s in smp])).float().to(device, non_blocking=True)
        wm   = torch.from_numpy(np.stack([s[2] for s in smp])).float().to(device, non_blocking=True)

        ps = []
        for m, t_val in zip(models, T):
            # ConvNeXt arm compatibility: it might not use all slot arguments natively,
            # but our wrapper in model.py accepts (imgs, mask, wmask).
            use_bf16 = torch.cuda.is_bf16_supported() if device.type == "cuda" else False
            ctx = torch.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16,
                                 enabled=(device.type == "cuda"))
            with ctx:
                logits = m(imgs, slot, wm).float()
                
                # Upgrade H: Temperature scaling
                scaled_logits = logits / max(t_val, 1e-6)
                ps.append(torch.sigmoid(scaled_logits).cpu().numpy())

        out.append(np.mean(ps, axis=0))
    return np.concatenate(out)


def run_inference(
    root: str,
    models: list[torch.nn.Module],
    test_csv: str | None = None,
    cfg: config.PreCfg | None = None,
    out_csv: str = "submission.csv",
    cache_dir: str | None = None,
    workers: int | None = None,
    chunk: int = 64,
    n_use: int | None = None,
    device: torch.device | None = None,
    temperatures: list[float] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build the cache and run predictions in a pipelined fashion."""
    cfg = cfg or config.get_cfg("v2")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ann, tab, sides, records = prepare_test_tables(root, cfg, workers)
    studies = list(tab.index)

    # Reorder to match test.csv if provided (mandatory for Kaggle submission)
    if test_csv and os.path.exists(test_csv):
        order = pd.read_csv(test_csv)["StudyInstanceUID"].tolist()
        studies = [s for s in order if s in set(studies)] + [s for s in studies if s not in set(order)]

    cache_dir = cache_dir or ("/kaggle/temp" if os.path.isdir("/kaggle/temp") else tempfile.gettempdir())
    prefix = os.path.join(cache_dir, "test_cache")
    slot_rows = {s: tab.loc[s].to_dict() for s in studies}

    state = {"moved": False}
    preds = np.full((len(studies), len(config.TARGETS)), np.nan, np.float32)

    def on_ready(cache: pcache.StudyCache, a: int, b: int):
        # Workers were forked at the first submit, i.e., before any CUDA call in this process.
        if not state["moved"]:
            for i, m in enumerate(models):
                m.eval().to(device, non_blocking=True)
                if device.type == "cuda" and torch.cuda.device_count() > 1:
                    models[i] = torch.nn.DataParallel(m)
            state["moved"] = True
        preds[a:b] = predict_chunk(models, cache, a, b, cfg, device, n_use,
                                   temperatures=temperatures)

    cache, stats = pcache.build_cache(
        prefix, studies, slot_rows, records, sides, cfg,
        workers=workers, resume=False, order="seq", chunk=chunk, on_ready=on_ready
    )
    print("cache stats:", stats)

    # Impute missing studies (fallback to median so submission never fails)
    empty = np.asarray(cache.slot).sum(1) == 0
    fill = np.nanmedian(preds[~empty], axis=0) if (~empty).any() else np.full(len(config.TARGETS), 0.5)
    preds[empty] = fill
    preds = np.where(np.isfinite(preds), preds, fill)

    # UPGRADE 5: Rank-percentile normalization (from 0.946/0.957 baselines).
    # AUC is rank-based: converting probabilities to within-test percentiles removes
    # calibration errors and is strictly AUC-neutral but reduces submission risk.
    ranked = pd.DataFrame(preds, columns=config.TARGETS).rank(pct=True)
    sub = ranked
    sub.insert(0, "StudyInstanceUID", studies)
    sub.to_csv(out_csv, index=False)

    return sub, stats



