"""One code path for everything: offline cache build, training fallback, and the inference notebook.

prepare_series / prepare_study NEVER raise (hidden-rerun lesson: data-dependent aborts are what broke earlier
public versions). Every problem becomes an entry in `info` and a masked slot, never a crash or a NaN.
"""
import numpy as np

import src.core.config as config
from . import geometry as geo
from . import pixels


def index_to_records(df):
    """DataFrame -> {SeriesInstanceUID: dict}. Done once; avoids pandas in the hot path."""
    return {r['SeriesInstanceUID']: r for r in df.to_dict('records')} if len(df) else {}


def study_sides(index_df):
    """{study: laterality dict} using the DICOM tag first, patient-x geometry second."""
    if index_df.empty or 'StudyInstanceUID' not in index_df.columns:
        return {}
    out = {}
    cols = [c for c in ('StudyInstanceUID', 'Laterality', 'ImageLaterality', 'center_x') if c in index_df.columns]
    for st, g in index_df[cols].groupby('StudyInstanceUID', sort=False):
        out[st] = geo.resolve_laterality(g.to_dict('records'))
    return out


def prepare_series(rec, plane, side, cfg):
    """-> (uint8 [D,S,S], valid [D], info) or (None, None, info)."""
    stack, valid, info = pixels.build_stack(rec, cfg)
    if stack is None:
        return None, None, info
    info['reoriented'] = False
    if cfg.reorient:
        stack, changed = geo.reorient_stack(stack, rec.get('orient_code'), plane)
        info['reoriented'] = bool(changed)          # (a transposition leaves the depth axis untouched)
    if cfg.lat_canon and side == 'R':
        stack = geo.canonical_side(stack, plane, 'R')
        if plane == 'Sagittal':
            valid = valid[::-1].copy()
        info['mirrored'] = True
    return stack, valid, info


def prepare_study(study, slot_row, records, side, cfg):
    """slot_row: {slot_name: SeriesInstanceUID or None}.

    -> dict(stack uint8 [S,D,H,W], valid bool [S,D], slot_mask uint8 [S], info {slot: info}, side)"""
    S, D, H = config.N_SLOTS, cfg.stack_depth, cfg.img_size
    stack = np.zeros((S, D, H, H), np.uint8)
    valid = np.zeros((S, D), bool)
    mask = np.zeros(S, np.uint8)
    infos = {}
    for k, (name, plane, _, _) in enumerate(config.SLOTS):
        uid = slot_row.get(name)
        if not uid or uid not in records:
            continue
        try:
            st, v, info = prepare_series(records[uid], plane, side, cfg)
        except Exception as e:                                   # belt and braces
            st, v, info = None, None, dict(errors=[f'{type(e).__name__}: {str(e)[:60]}'])
        infos[name] = info
        if st is None:
            continue
        stack[k], valid[k], mask[k] = st, v, 1
    return dict(stack=stack, valid=valid, slot_mask=mask, info=infos, side=side)
