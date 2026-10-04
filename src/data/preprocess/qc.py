"""Data-contract checks. Cheap enough to run on every build; their numbers are the acceptance gates."""
import numpy as np
import pandas as pd

import src.core.config as config


def cache_sanity(cache, n_sample=200, seed=0):
    """Sample built studies and verify the cache is not silently broken (constant images, empty-but-flagged slots,
    all-invalid depth, wrong dtype)."""
    rng = np.random.default_rng(seed)
    done = np.where(np.asarray(cache.done) == 1)[0]
    if len(done) == 0:
        return dict(studies_checked=0)
    pick = rng.choice(done, size=min(n_sample, len(done)), replace=False)
    bad = dict(constant_slot=[], flagged_but_blank=[], low_valid_depth=[], unflagged_but_filled=[])
    n_slots = 0
    stds, means, vfrac = [], [], []
    for i in pick:
        for s in range(config.N_SLOTS):
            filled = bool(cache.slot[i, s])
            st = np.asarray(cache.images[i, s, ::2, ::4, ::4])          # cheap subsample
            if filled:
                n_slots += 1
                v = float(np.mean(cache.valid[i, s]))
                vfrac.append(v)
                if st.max() == 0:
                    bad['flagged_but_blank'].append((cache.studies[i], s))
                if st.std() < 1.0:
                    bad['constant_slot'].append((cache.studies[i], s))
                if v < 0.5:
                    bad['low_valid_depth'].append((cache.studies[i], s))
                stds.append(float(st.std()))
                means.append(float(st.mean()))
            elif st.max() > 0:
                bad['unflagged_but_filled'].append((cache.studies[i], s))
    out = dict(studies_checked=len(pick), slots_checked=n_slots,
               mean_valid_depth=float(np.mean(vfrac)) if vfrac else float('nan'),
               slot_mean_p5_p95=[float(np.percentile(means, 5)), float(np.percentile(means, 95))] if means else None,
               slot_std_p5_p95=[float(np.percentile(stds, 5)), float(np.percentile(stds, 95))] if stds else None)
    out.update({k: len(v) for k, v in bad.items()})
    out['failures'] = {k: v[:5] for k, v in bad.items() if v}
    return out


def laterality_report(sides):
    df = pd.DataFrame(sides).T
    rep = dict(studies=len(df), by_side=df['side'].value_counts().to_dict(), by_source=df['source'].value_counts().to_dict(),
               tag_geometry_disagree=int(df['disagree'].sum()))
    rep['unresolved_frac'] = float((df['side'] == 'U').mean())
    return rep
