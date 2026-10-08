"""Calibration: measure real throughput on a SMALL sample of the actual dataset, then project the full cost.

Run it first in a fresh Kaggle session (takes ~1-3 minutes): it tells you how long the index scan and the cache build
will take on THIS machine/filesystem, how much scratch disk the cache needs, and whether the training loop's data
loading can keep a GPU fed.
"""
import os
import time

import numpy as np
import pandas as pd

import src.core.config as config
from . import index as pix
from . import slots as pslots
from . import pipeline
from . import cache as pcache
from . import loader
from . import runner


def _items_for(root, split, studies):
    base = os.path.join(root, pix.SPLIT_DIRS[split])
    items = []
    for st in studies:
        for e in sorted(os.scandir(os.path.join(base, st)), key=lambda x: x.name):
            if e.is_dir():
                items.append((split, st, e.name, e.path))
    return items


def calibrate(root, split='train', n_studies=24, workers=None, preset='v2', scratch=None, step_seconds=1.0,
              batch_size=config.BATCH_SIZE, n_use=4):
    root = config.resolve_data_root(root)
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    cfg = config.get_cfg(preset)
    workers = workers or max(2, (os.cpu_count() or 4))
    scratch = scratch or runner.pick_scratch(verbose=False)
    base = os.path.join(root, pix.SPLIT_DIRS[split])
    studies = sorted(e.name for e in os.scandir(base) if e.is_dir())[:n_studies]
    items = _items_for(root, split, studies)
    n_total_series = n_total_studies = None
    for name, attr in (('train_series.csv' if split == 'train' else 'test_series.csv', 'series'), ('train.csv' if split == 'train' else 'test.csv', 'studies')):
        p = os.path.join(root, name)
        if os.path.exists(p):
            n = len(pd.read_csv(p, usecols=[0]))
            if attr == 'series':
                n_total_series = n
            else:
                n_total_studies = n
    n_total_series = n_total_series or len(items) * 4407 // max(len(studies), 1)
    n_total_studies = n_total_studies or 4407
    res = {}
    # ---- 1. index scan
    t = time.time()
    try:
        ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'))
        rows = list(ex.map(pix.scan_series, items, chunksize=4))
        ex.shutdown()
    except Exception:
        rows = [pix.scan_series(i) for i in items]
    dt = time.time() - t
    df = pix.attach_csv_flags(pd.DataFrame(rows), root)
    files = int(df['n_files'].sum())
    res['index_s_per_series'] = dt / len(items)
    res['index_files_per_s'] = files / dt
    res['index_minutes_full'] = res['index_s_per_series'] * n_total_series / 60
    # ---- 2. cache build
    ann, tab = pslots.assign_all(df, cfg.slot_prefer_2d, cfg.slot_fs_priority, cfg.slot_csv_fallback)
    sides = pipeline.study_sides(df)
    records = pipeline.index_to_records(ann)
    slot_rows = {s: tab.loc[s].to_dict() for s in studies if s in tab.index}
    prefix = os.path.join(scratch, 'calib_cache')
    cache, st = pcache.build_cache(prefix, studies, slot_rows, records, sides, cfg, workers, resume=False, progress=False)
    res['cache_s_per_study_wall'] = st['seconds'] / len(studies)
    res['cache_cpu_s_per_study'] = st['seconds'] * workers / len(studies)
    res['cache_minutes_full'] = res['cache_s_per_study_wall'] * n_total_studies / 60
    res['cache_gb_full'] = runner.estimate_cache_gb(n_total_studies, cfg)
    # ---- 3. training loop data rate (single process, with augmentation)
    rng = np.random.default_rng(0)
    idx = [i for i in range(len(studies)) if cache.slot[i].any()]
    t = time.time()
    reps = 3
    for _ in range(reps):
        for i in idx:
            loader.make_sample(cache, i, cfg, True, n_use, rng, True)
    per = (time.time() - t) / (reps * len(idx))
    res['loader_ms_per_sample_1proc'] = per * 1000
    res['loader_samples_per_s_1proc'] = 1.0 / per
    need = batch_size / step_seconds
    res['loader_needed_samples_per_s'] = need
    res['loader_headroom_x'] = (1.0 / per) * min(4, workers) / need
    free = pcache.shutil.disk_usage(scratch).free / 1e9
    res['scratch_free_gb'] = free
    res['studies_sampled'] = len(studies)
    res['series_sampled'] = len(items)
    for k in ('calib_cache.images.u8', 'calib_cache.valid.u1', 'calib_cache.slot.u1', 'calib_cache.done.u1', 'calib_cache.meta.json', 'calib_cache.meta.csv'):
        try:
            os.remove(os.path.join(scratch, k))
        except OSError:
            pass
    print(f"sample: {len(studies)} studies / {len(items)} series / {files} slices on {workers} workers (preset {cfg.name})")
    print(f"  index scan   : {res['index_files_per_s']:.0f} files/s  -> full train scan ~ {res['index_minutes_full']:.1f} min")
    print(f"  cache build  : {res['cache_s_per_study_wall']:.2f} s/study wall ({res['cache_cpu_s_per_study']:.1f} cpu-s)"
          f"  -> full train cache ~ {res['cache_minutes_full']:.0f} min, {res['cache_gb_full']:.1f} GB (scratch free {free:.0f} GB)")
    print(f"  train loader : {res['loader_ms_per_sample_1proc']:.0f} ms/sample/proc -> {res['loader_headroom_x']:.1f}x headroom "
          f"over the {need:.0f} samples/s a {step_seconds:.1f}s step with batch {batch_size} needs (4 workers)")
    if res['cache_gb_full'] > free * 0.92:
        print('  !! cache does not fit the scratch disk: use runner.fit_cache_cfg(...) to pick a smaller img_size/stack_depth')
    return res
