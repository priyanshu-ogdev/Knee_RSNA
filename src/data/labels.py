"""Label table for training: gold labels + optional extra (report-derived / pseudo) labels with confidence weights.

Contract consumed by RSNADataset: StudyInstanceUID, the 12 TARGETS (NaN = unlabeled -> masked), optional
'<target>_weight' columns. 98.7% of train studies have no gold label, so the extra table is where most supervision
comes from; gold always wins where it exists.
"""
import os
import numpy as np
import pandas as pd

import src.core.config as config


def build_labels(root, extra_csv=None, extra_weight=0.5, out_csv=None):
    """root: competition folder. extra_csv: same schema as train.csv (targets may be soft probabilities in [0,1]),
    optionally with '<target>_weight' columns (extractor confidence). -> DataFrame (and CSV if out_csv)."""
    tr = pd.read_csv(os.path.join(root, 'train.csv'))
    out = pd.DataFrame({'StudyInstanceUID': tr['StudyInstanceUID'].astype(str).str.strip()})
    gold = {t: (tr[t].astype(float) if t in tr.columns else pd.Series(np.nan, index=tr.index)) for t in config.TARGETS}
    ex = None
    if extra_csv and os.path.exists(extra_csv):
        ex_raw = pd.read_csv(extra_csv)
        ex_raw['StudyInstanceUID'] = ex_raw['StudyInstanceUID'].astype(str).str.strip()
        ex = ex_raw.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID').reindex(out['StudyInstanceUID'])
    out['source'] = 'none'
    any_gold = pd.concat(gold, axis=1).notna().any(axis=1).values
    out.loc[any_gold, 'source'] = 'gold'
    for t in config.TARGETS:
        g = gold[t].values
        y = g.copy()
        w = np.where(np.isfinite(g), 1.0, 0.0)
        if ex is not None and t in ex.columns:
            e = ex[t].astype(float).values
            ew = ex[f'{t}_weight'].astype(float).fillna(1.0).values if f'{t}_weight' in ex.columns else 1.0
            use = ~np.isfinite(g) & np.isfinite(e)
            y = np.where(use, e, g)
            w = np.where(use, extra_weight * ew, w)
            out.loc[use & (out['source'] == 'none'), 'source'] = 'extra'
        out[t] = y
        out[f'{t}_weight'] = w
    if out_csv:
        out.to_csv(out_csv, index=False)
    return out
