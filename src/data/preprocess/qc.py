"""Data-contract checks. Cheap enough to run on every build; their numbers are the acceptance gates."""
import numpy as np
import pandas as pd

import src.core.config as config


def cache_sanity(cache, n_sample=200, seed=0, expected_slot_mask=None):
    """Sample built studies and verify the cache is not silently broken (constant images, empty-but-flagged slots,
    all-invalid depth, wrong dtype)."""
    rng = np.random.default_rng(seed)
    done = np.where(np.asarray(cache.done) == 1)[0]
    if len(done) == 0:
        return dict(studies_checked=0, accepted=False, reason="no completed studies")
    not_done = np.where(np.asarray(cache.done) != 1)[0]
    failed = np.asarray(cache.failed)
    decode_errors = np.asarray(cache.errors)
    slot_array = np.asarray(cache.slot)
    done_array = np.asarray(cache.done)
    valid_array = np.asarray(cache.valid)
    expected_failures = []
    if expected_slot_mask is not None:
        expected_slot_mask = np.asarray(expected_slot_mask, dtype=bool)
        if expected_slot_mask.shape != cache.slot.shape:
            raise ValueError(
                f"Expected slot mask shape {expected_slot_mask.shape} does not match cache {cache.slot.shape}"
            )
        expected_failures = np.argwhere(expected_slot_mask & (np.asarray(cache.slot) == 0))
    empty_studies = int((slot_array.sum(axis=1) == 0).sum())
    invalid_done_flags = int((~np.isin(done_array, [0, 1])).sum())
    invalid_slot_flags = int((~np.isin(slot_array, [0, 1])).sum())
    invalid_valid_flags = int((~np.isin(valid_array, [0, 1])).sum())
    unflagged_valid_slots = int(
        ((slot_array == 0) & (valid_array.sum(axis=2) > 0)).sum()
    )
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
               total_studies=len(cache.studies),
               incomplete_studies=len(not_done),
               empty_studies=empty_studies,
               invalid_done_flags=invalid_done_flags,
               invalid_slot_flags=invalid_slot_flags,
               invalid_valid_flags=invalid_valid_flags,
               unflagged_valid_slots=unflagged_valid_slots,
               decode_errors=int(decode_errors.sum()),
               failed_selected_slots=int(failed.sum()),
               missing_expected_slots=len(expected_failures),
               mean_valid_depth=float(np.mean(vfrac)) if vfrac else float('nan'),
               slot_mean_p5_p95=[float(np.percentile(means, 5)), float(np.percentile(means, 95))] if means else None,
               slot_std_p5_p95=[float(np.percentile(stds, 5)), float(np.percentile(stds, 95))] if stds else None)
    out.update({k: len(v) for k, v in bad.items()})
    out['failures'] = {k: v[:5] for k, v in bad.items() if v}
    out['expected_slot_failure_examples'] = [
        (cache.studies[int(i)], int(s)) for i, s in expected_failures[:10]
    ]
    blocking_pixel_errors = any(
        bad[key]
        for key in ("constant_slot", "flagged_but_blank", "unflagged_but_filled")
    )
    out['accepted'] = bool(
        len(not_done) == 0
        and empty_studies == 0
        and invalid_done_flags == 0
        and invalid_slot_flags == 0
        and invalid_valid_flags == 0
        and unflagged_valid_slots == 0
        and decode_errors.sum() == 0
        and failed.sum() == 0
        and len(expected_failures) == 0
        and not blocking_pixel_errors
        and out['mean_valid_depth'] > 0.8
    )
    return out


def laterality_report(sides):
    if not sides:
        return dict(studies=0, by_side={}, by_source={}, tag_geometry_disagree=0, unresolved_frac=0.0)
    df = pd.DataFrame(sides).T
    rep = dict(studies=len(df), by_side=df['side'].value_counts().to_dict() if 'side' in df else {},
               by_source=df['source'].value_counts().to_dict() if 'source' in df else {},
               tag_geometry_disagree=int(df['disagree'].sum()) if 'disagree' in df else 0)
    rep['unresolved_frac'] = float((df['side'] == 'U').mean()) if 'side' in df else 0.0
    return rep
