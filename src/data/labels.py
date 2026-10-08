"""Label table for training: gold labels + optional extra (report-derived / pseudo) labels with confidence weights.

Contract consumed by RSNADataset: StudyInstanceUID, the 12 TARGETS (NaN = unlabeled -> masked), optional
'<target>_weight' columns. 98.7% of train studies have no gold label, so the extra table is where most supervision
comes from; gold always wins where it exists.
"""
import os
import hashlib
import numpy as np
import pandas as pd

import src.core.config as config


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


from src.core.config import resolve_data_root

def build_labels(root, extra_csv=None, extra_weight=0.5, out_csv=None):
    root = resolve_data_root(root)
    """root: competition folder. extra_csv: same schema as train.csv (targets may be soft probabilities in [0,1]),
    optionally with '<target>_weight' columns (extractor confidence). -> DataFrame (and CSV if out_csv)."""
    if not np.isfinite(extra_weight) or not 0.0 <= extra_weight <= 1.0:
        raise ValueError(f"extra_weight must be finite and in [0, 1], got {extra_weight!r}")
    tr = pd.read_csv(os.path.join(root, 'train.csv'))
    if 'StudyInstanceUID' not in tr or tr['StudyInstanceUID'].isna().any():
        raise ValueError("train.csv contains a missing StudyInstanceUID")
    out = pd.DataFrame({'StudyInstanceUID': tr['StudyInstanceUID'].astype(str).str.strip()})
    if out['StudyInstanceUID'].eq('').any() or out['StudyInstanceUID'].duplicated().any():
        raise ValueError("train.csv must have non-empty, unique StudyInstanceUID values")
    gold = {t: (tr[t].astype(float) if t in tr.columns else pd.Series(np.nan, index=tr.index)) for t in config.TARGETS}
    for target, values in gold.items():
        if np.isinf(values.to_numpy(dtype=float)).any():
            raise ValueError(f"gold labels for {target!r} must be finite or missing")
        finite = values[np.isfinite(values)]
        if not finite.between(0.0, 1.0).all():
            raise ValueError(f"gold labels for {target!r} must be in [0, 1]")
    ex = None
    if extra_csv and os.path.exists(extra_csv):
        ex_raw = pd.read_csv(extra_csv)
        if 'StudyInstanceUID' not in ex_raw.columns or ex_raw['StudyInstanceUID'].isna().any():
            raise ValueError(f"extra labels must contain non-null StudyInstanceUID values: {extra_csv}")
        ex_raw['StudyInstanceUID'] = ex_raw['StudyInstanceUID'].astype(str).str.strip()
        if ex_raw['StudyInstanceUID'].eq('').any() or ex_raw['StudyInstanceUID'].duplicated().any():
            raise ValueError(f"extra labels must have non-empty, unique StudyInstanceUID values: {extra_csv}")
        unknown = sorted(set(ex_raw['StudyInstanceUID']) - set(out['StudyInstanceUID']))
        if unknown:
            raise ValueError(
                f"extra labels contain {len(unknown)} StudyInstanceUID values absent from train.csv; "
                f"examples: {unknown[:5]}"
            )
        ex = ex_raw.set_index('StudyInstanceUID').reindex(out['StudyInstanceUID'])
    out['source'] = 'none'
    any_gold = pd.concat(gold, axis=1).notna().any(axis=1).values
    out.loc[any_gold, 'source'] = 'gold'
    for t in config.TARGETS:
        g = gold[t].values
        y = g.copy()
        w = np.where(np.isfinite(g), 1.0, 0.0)
        if ex is not None and t in ex.columns:
            e = ex[t].astype(float).values
            
            # SOTA Calibration: Estimate P(positive | not_stated) using Gold Studies
            is_soft_neg = (e == -1.0)
            if is_soft_neg.any():
                gold_valid = g[np.isfinite(g)]
                if len(gold_valid) > 0:
                    gold_prevalence = np.mean(gold_valid)
                else:
                    gold_prevalence = 0.05
                calibrated_prob = min(0.15, gold_prevalence * 0.8) # Conservative penalty
                e[is_soft_neg] = calibrated_prob
            
            if np.isinf(e).any():
                raise ValueError(f"extra labels for {t!r} must be finite or missing")
            finite = e[np.isfinite(e)]
            if not np.logical_and(finite >= 0.0, finite <= 1.0).all():
                raise ValueError(f"extra labels for {t!r} must be in [0, 1]")
            ew = ex[f'{t}_weight'].astype(float).fillna(1.0).values if f'{t}_weight' in ex.columns else 1.0
            if np.isscalar(ew):
                pass
            elif not np.isfinite(ew).all() or (ew < 0).any() or (ew > 1).any():
                raise ValueError(f"extra confidence weights for {t!r} must be finite and in [0, 1]")
            use = ~np.isfinite(g) & np.isfinite(e)
            y = np.where(use, e, g)
            w = np.where(use, extra_weight * ew, w)
            out.loc[use & (out['source'] == 'none'), 'source'] = 'extra'
        out[t] = y
        out[f'{t}_weight'] = w
    if out_csv:
        temporary = f"{out_csv}.tmp"
        out.to_csv(temporary, index=False)
        os.replace(temporary, out_csv)
        manifest = {
            "schema_version": "train-labels-v2",
            "train_csv_sha256": _sha256_file(os.path.join(root, 'train.csv')),
            "extra_csv_sha256": (
                _sha256_file(extra_csv)
                if extra_csv and os.path.exists(extra_csv)
                else None
            ),
            "labels_csv_sha256": _sha256_file(out_csv),
            "studies": len(out),
            "source_counts": out['source'].value_counts().to_dict(),
            "per_target": {
                t: {
                    "labeled": int(out[t].notna().sum()),
                    "full_weight": int((out[f'{t}_weight'] == 1.0).sum()),
                    "positive": int((out[t] > 0).sum()),
                }
                for t in config.TARGETS
            },
        }
        manifest_path = f"{out_csv}.manifest.json"
        manifest_tmp = f"{manifest_path}.tmp"
        with open(manifest_tmp, "w", encoding="utf-8") as stream:
            import json
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(manifest_tmp, manifest_path)
    return out
