"""Series index: one row per series on disk, with anatomical slice order, geometry and a representative header.

Design rules (from the dataset audit + the hidden-rerun failures documented in the public notebooks):
  * the DIRECTORY TREE is the truth for which series/files exist; the CSVs only contribute flags
    (test_series.csv in the hidden set may differ from the placeholder -> never assume it is complete)
  * a failure inside one series never aborts the run: the row is kept with `err` set and an empty file list
  * the build is chunked + checkpointed, so a 40-minute scan survives a session restart
"""
import os
import time
import pickle
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd

import src.core.config as config
from . import dicomio as dio
from . import geometry as geo

SPLIT_DIRS = {'train': 'train_series', 'test': 'test_series'}


def discover_root(explicit=None):
    """Find the competition folder (handles /kaggle/input/competitions/<slug> and /kaggle/input/<slug>)."""
    cands = [explicit, os.environ.get('KNEE_DATA')] + list(config.ROOT_CANDIDATES)
    for c in cands:
        if c and os.path.isdir(c) and (os.path.isdir(os.path.join(c, 'train_series')) or
                                       os.path.isdir(os.path.join(c, 'test_series'))):
            return c
    base = '/kaggle/input'
    if os.path.isdir(base):
        for p in sorted(os.listdir(base)):
            for q in (os.path.join(base, p), *[os.path.join(base, p, x) for x in sorted(os.listdir(os.path.join(base, p)))[:20]
                                                 if os.path.isdir(os.path.join(base, p, x))]):
                if os.path.isdir(os.path.join(q, 'test_series')) or os.path.isdir(os.path.join(q, 'train_series')):
                    return q
    raise FileNotFoundError('competition root not found; pass it explicitly or set KNEE_DATA')


def list_series_dirs(root, split):
    base = os.path.join(root, SPLIT_DIRS[split])
    items = []
    if not os.path.isdir(base):
        return items
    with os.scandir(base) as it:
        studies = sorted(e.name for e in it if e.is_dir())
    for st in studies:
        sd = os.path.join(base, st)
        with os.scandir(sd) as it2:
            for e in sorted(it2, key=lambda x: x.name):
                if e.is_dir():
                    items.append((split, st, e.name, e.path))
    return items


def scan_series(item):
    """Full slice ordering + representative header for one series directory."""
    split, study, series, d = item
    row = dict(split=split, StudyInstanceUID=study, SeriesInstanceUID=series, dir=d, n_files=0,
               ordered_files=[], positions=np.zeros(0), err=None)
    try:
        names = dio.list_dicoms(d)
        row['n_files'] = len(names)
        recs, bad = [], 0
        for nm in names:
            ds = dio.read_tags(os.path.join(d, nm), dio.ORDER_TAGS)
            if ds is None or 'Rows' not in ds or 'Columns' not in ds:
                bad += 1
                continue
            ps = dio.floats(getattr(ds, 'PixelSpacing', None), 2)
            recs.append(dict(name=nm, ipp=dio.floats(getattr(ds, 'ImagePositionPatient', None), 3),
                             iop=dio.floats(getattr(ds, 'ImageOrientationPatient', None), 6),
                             inst=dio.f1(getattr(ds, 'InstanceNumber', None)),
                             rows=int(ds.Rows), cols=int(ds.Columns), ps=tuple(ps) if ps else None))
        row['n_bad'] = bad
        o = geo.order_records(recs, 'position')
        if o is None:
            row['err'] = 'no_readable_slices'
            return row
        row.update(ordered_files=o['ordered'], positions=o['pos'].astype(np.float32), order_method=o['method'],
                   n_dropped_shape=o['n_dropped_shape'], plane_geo=o['plane'], orient_code=o['code'],
                   plane_align=o['align'], spacing_med=o['spacing_med'], z_extent=o['z_extent'],
                   rows=o['shape'][0], cols=o['shape'][1], ps_row=o['ps'][0], ps_col=o['ps'][1],
                   center_x=o['center_x'], nsign=o['nsign'], n_slices=len(o['ordered']))
        rep = dio.representative_header(os.path.join(d, o['ordered'][o['rep_index']]))
        if rep:
            row.update({k: v for k, v in rep.items() if k not in ('ps_row', 'ps_col')})
            if not np.isfinite(row['ps_row']) and np.isfinite(rep['ps_row']):
                row['ps_row'], row['ps_col'] = rep['ps_row'], rep['ps_col']
    except Exception as e:                                     # never fatal
        row['err'] = f'{type(e).__name__}: {str(e)[:80]}'
    return row


def build_index(root, splits=('train', 'test'), workers=None, chunk=1000, cache_dir=None, limit=0, force=False,
                progress=True):
    """Scan every series on disk. Returns a DataFrame (object columns: ordered_files, positions)."""
    workers = workers or max(2, (os.cpu_count() or 4) * 2)
    items = []
    for sp in splits:
        items += list_series_dirs(root, sp)
    if limit:
        items = items[:limit]
    chunks = [items[i:i + chunk] for i in range(0, len(items), chunk)]
    if cache_dir:                                    # stale-checkpoint guard: chunks are only valid for THIS item list
        import glob
        import hashlib
        sig = hashlib.md5(('|'.join(i[3] for i in items) + f'#{chunk}').encode()).hexdigest()
        sp = os.path.join(cache_dir, 'signature.txt')
        os.makedirs(cache_dir, exist_ok=True)
        if not (os.path.exists(sp) and open(sp).read() == sig):
            for f in glob.glob(os.path.join(cache_dir, 'index_*.pkl')):
                os.remove(f)
            open(sp, 'w').write(sig)
    parts, t0 = [], time.time()
    ex = None
    try:
        ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'))
    except Exception:
        ex = ThreadPoolExecutor(max_workers=workers)
    for k, ch in enumerate(chunks):
        p = os.path.join(cache_dir, f'index_{k:04d}.pkl') if cache_dir else None
        if p and os.path.exists(p) and not force:
            parts.append(pd.read_pickle(p))
            continue
        df = pd.DataFrame(list(ex.map(scan_series, ch, chunksize=8)))
        if p:
            os.makedirs(cache_dir, exist_ok=True)
            df.to_pickle(p)
        parts.append(df)
        if progress:
            print(f'  index chunk {k + 1}/{len(chunks)}  {(k + 1) * chunk / max(time.time() - t0, 1e-9):.1f} series/s', flush=True)
    ex.shutdown()
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out


def attach_csv_flags(index_df, root):
    """Left-join the competition CSV flags (plane / fluid / fat-sat) onto the on-disk index."""
    if index_df.empty or 'SeriesInstanceUID' not in index_df.columns:
        for c in ('SeriesInstanceUID', 'Anatomical_Plane', 'Fluid_Sensitive', 'Fat_Suppression'):
            if c not in index_df.columns:
                index_df[c] = [] if index_df.empty else (np.nan if c != 'Anatomical_Plane' else None)
        return index_df
    frames = []
    for name in ('train_series.csv', 'test_series.csv'):
        p = os.path.join(root, name)
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        for c in ('Anatomical_Plane', 'Fluid_Sensitive', 'Fat_Suppression'):
            index_df[c] = np.nan if c != 'Anatomical_Plane' else None
        return index_df
    csv = pd.concat(frames, ignore_index=True).drop_duplicates('SeriesInstanceUID')
    keep = [c for c in ('SeriesInstanceUID', 'Anatomical_Plane', 'Fluid_Sensitive', 'Fat_Suppression') if c in csv.columns]
    out = index_df.merge(csv[keep], on='SeriesInstanceUID', how='left')
    for c in ('Anatomical_Plane', 'Fluid_Sensitive', 'Fat_Suppression'):
        if c not in out.columns:
            out[c] = np.nan if c != 'Anatomical_Plane' else None
    return out


def save_index(df, path):
    with open(path, 'wb') as fh:
        pickle.dump(df, fh, protocol=4)


def load_index(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


def index_report(df):
    """Human-readable data-contract statistics (printed by tools/build_index.py)."""
    rep = {}
    if df.empty:
        rep['series'] = 0
        rep['series_with_error'] = 0
        rep['empty_series'] = 0
        return rep
    ok = df[df.err.isna()] if 'err' in df else df
    rep['series'] = len(df)
    rep['series_with_error'] = int(df.err.notna().sum()) if 'err' in df else 0
    rep['empty_series'] = int((df.n_files == 0).sum()) if 'n_files' in df else 0
    if 'order_method' in ok:
        rep['order_method'] = ok.order_method.value_counts().to_dict()
    if 'orient_code' in ok and 'plane_geo' in ok:
        rep['orient_codes'] = {f'{p}:{c}': int(n) for (p, c), n in ok.groupby('plane_geo').orient_code.value_counts().items()}
        bad = 0
        for pl, code in config.CANON_ORIENT.items():
            sub = ok[ok.plane_geo == pl]
            bad += int((sub.orient_code != code).sum())
        rep['non_canonical_orientation'] = bad
    if 'Anatomical_Plane' in ok:
        m = ok[ok.Anatomical_Plane.notna() & ok.plane_geo.notna()]
        rep['csv_plane_vs_geometry_mismatch'] = int((m.Anatomical_Plane != m.plane_geo).sum())
    if 'n_dropped_shape' in ok:
        rep['series_with_mixed_shapes'] = int((ok.n_dropped_shape > 0).sum())
    if 'n_bad' in ok:
        rep['series_with_unreadable_slices'] = int((ok.n_bad > 0).sum())
    return rep
