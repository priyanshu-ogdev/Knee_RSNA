"""Pipelined inference: CPU workers build the cache for chunk k+1 while the GPU predicts chunk k.

Hidden-rerun rules (the public notebooks' failure history): nothing here may raise on data conditions. Missing
series/flags/metadata degrade to masked slots; studies that end up with no usable slot fall back to the median
prediction of the other studies instead of a constant column; the submission is always written.
"""
import os
import tempfile
import numpy as np
import pandas as pd
import torch

from . import config
from .preprocess import index as pix
from .preprocess import slots as pslots
from .preprocess import pipeline, cache as pcache, loader


def rank_ensemble(dino_df, coatnet_df):
    """
    0.943 SOTA Rank Ensembling logic. Converts raw probabilities to percentiles (ranks),
    then applies target-specific blending weights.
    """
    coatnet_df = coatnet_df.set_index('StudyInstanceUID').reindex(dino_df['StudyInstanceUID']).reset_index()
    dino_ranks = dino_df[config.TARGETS].rank(method='average', pct=True)
    coat_ranks = coatnet_df[config.TARGETS].rank(method='average', pct=True)

    coat_weights = {label: 0.60 for label in config.TARGETS}
    coat_weights.update({
        'ACL': 0.75,
        'Medial Meniscus': 0.80,
        'Lateral Meniscus': 1.00,
        'Lateral OA': 0.75,
        'Fracture': 0.75,
    })

    blend_df = dino_df.copy()
    for label in config.TARGETS:
        w = coat_weights[label]
        blend_df[label] = ((1.0 - w) * dino_ranks[label]) + (w * coat_ranks[label])

    blend_df[config.TARGETS] = blend_df[config.TARGETS].rank(method='average', pct=True)
    return blend_df


def prepare_test_tables(root, cfg, workers=None):
    """Directory-truth index of test_series -> annotated series table, slot table, laterality, records."""
    idx = pix.build_index(root, ('test',), workers=workers, chunk=500, progress=False)
    idx = pix.attach_csv_flags(idx, root)
    idx['n_slices'] = idx['n_slices'].fillna(0) if 'n_slices' in idx else 0
    ann, tab = pslots.assign_all(idx, cfg.slot_prefer_2d, cfg.slot_fs_priority, cfg.slot_csv_fallback)
    return ann, tab, pipeline.study_sides(idx), pipeline.index_to_records(ann)


@torch.no_grad()
def predict_chunk(models, cache, a, b, cfg, device, n_use=None, batch=4):
    out = []
    for lo in range(a, b, batch):
        rows = list(range(lo, min(lo + batch, b)))
        smp = [loader.make_sample(cache, i, cfg, False, n_use) for i in rows]
        imgs = torch.from_numpy(np.stack([s[0] for s in smp])).to(device)
        slot = torch.from_numpy(np.stack([s[1] for s in smp])).float().to(device)
        wm = torch.from_numpy(np.stack([s[2] for s in smp])).float().to(device)
        ps = []
        for m in models:
            with torch.autocast('cuda', enabled=device.type == 'cuda'):
                ps.append(torch.sigmoid(m(imgs, slot, wm).float()).cpu().numpy())
        out.append(np.mean(ps, axis=0))
    return np.concatenate(out)


def run_inference(root, models, test_csv=None, cfg=None, out_csv='submission.csv', cache_dir=None, workers=None,
                  chunk=64, n_use=None, device=None):
    cfg = cfg or config.get_cfg('v2')
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ann, tab, sides, records = prepare_test_tables(root, cfg, workers)
    studies = list(tab.index)
    if test_csv and os.path.exists(test_csv):                           # keep the competition's study order
        order = pd.read_csv(test_csv)['StudyInstanceUID'].tolist()
        studies = [s for s in order if s in set(studies)] + [s for s in studies if s not in set(order)]
    cache_dir = cache_dir or ('/kaggle/temp' if os.path.isdir('/kaggle/temp') else tempfile.gettempdir())
    prefix = os.path.join(cache_dir, 'test_cache')
    slot_rows = {s: tab.loc[s].to_dict() for s in studies}
    state = {'moved': False}
    preds = np.full((len(studies), len(config.TARGETS)), np.nan, np.float32)

    def on_ready(cache, a, b):                                          # runs while workers build later chunks
        if not state['moved']:      # workers were forked at the first submit, i.e. before any CUDA call in this process
            for m in models:
                m.eval().to(device)
            state['moved'] = True
        preds[a:b] = predict_chunk(models, cache, a, b, cfg, device, n_use)

    cache, stats = pcache.build_cache(prefix, studies, slot_rows, records, sides, cfg, workers=workers, resume=False,
                                      order='seq', chunk=chunk, on_ready=on_ready)
    print('cache stats:', stats)
    empty = np.asarray(cache.slot).sum(1) == 0
    fill = np.nanmedian(preds[~empty], axis=0) if (~empty).any() else np.full(len(config.TARGETS), 0.5)
    preds[empty] = fill                                                 # never a constant column from empty studies
    preds = np.where(np.isfinite(preds), preds, fill)
    sub = pd.DataFrame(preds, columns=config.TARGETS)
    sub.insert(0, 'StudyInstanceUID', studies)
    sub.to_csv(out_csv, index=False)
    return sub, stats
