"""Memory-mapped uint8 study cache + a parallel builder with no inter-process array traffic.

Layout (one contiguous 16 MB block per study at D=24, S=6, 336^2 -> one sequential read per sample):
    <prefix>.images.u8   [N, S, D, H, W] uint8        <prefix>.valid.u1  [N, S, D]
    <prefix>.slot.u1     [N, S]                       <prefix>.done.u1   [N]
    <prefix>.meta.json   study ids, cfg, shapes       <prefix>.meta.csv  per-study side/source/qc

Throughput design:
  * workers write straight into the shared memmap (MAP_SHARED) -> nothing but a tiny dict is pickled back
  * longest-processing-time-first scheduling (3-D series are ~5x a 2-D one) removes the straggler tail
  * resumable: `done` flags live in a memmap, so a killed run restarts where it stopped
  * optional streaming hook (`on_ready`) lets inference run the GPU on chunk k while CPUs build chunk k+1
"""
import os
import json
import time
import shutil
import multiprocessing as mp
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

import src.core.config as config
from . import pipeline

_G = {}


def _paths(prefix):
    return {k: f'{prefix}.{v}' for k, v in dict(images='images.u8', valid='valid.u1', slot='slot.u1',
                                                 done='done.u1', meta='meta.json', csv='meta.csv').items()}


class StudyCache:
    def __init__(self, prefix, mode='r'):
        self.prefix, self.mode = prefix, mode
        p = _paths(prefix)
        with open(p['meta']) as fh:
            self.meta = json.load(fh)
        self.studies = self.meta['studies']
        N, S, D, H = self.meta['N'], self.meta['S'], self.meta['D'], self.meta['H']
        m = 'r+' if mode == 'r+' else 'r'
        self.images = np.memmap(p['images'], np.uint8, m, shape=(N, S, D, H, H))
        self.valid = np.memmap(p['valid'], np.uint8, m, shape=(N, S, D))
        self.slot = np.memmap(p['slot'], np.uint8, m, shape=(N, S))
        self.done = np.memmap(p['done'], np.uint8, m, shape=(N,))
        self.index = {s: i for i, s in enumerate(self.studies)}
        # LINUX / GB10 SPEEDUP: Hint kernel for standard cached access so RAM buffers active studies
        try:
            import mmap as _py_mmap
            # SOTA Memory Optimization: For read mode (training/workers), use MADV_RANDOM.
            # This prevents the Linux kernel page-cache from locking 65GB of disk pages in RAM.
            adv_flag = getattr(_py_mmap, 'MADV_RANDOM', getattr(_py_mmap, 'MADV_NORMAL', None)) if mode == 'r' else getattr(_py_mmap, 'MADV_NORMAL', None)
            if adv_flag is not None:
                for a in (self.images, self.valid, self.slot, self.done):
                    mm = getattr(a, '_mmap', None) or getattr(getattr(a, 'base', None), '_mmap', None)
                    if mm and hasattr(mm, 'madvise'):
                        mm.madvise(adv_flag)
        except Exception:
            pass

    def __len__(self):
        return len(self.studies)

    @staticmethod
    def create(prefix, studies, cfg):
        N, S, D, H = len(studies), config.N_SLOTS, cfg.stack_depth, cfg.img_size
        need = N * S * D * H * H
        d = os.path.dirname(os.path.abspath(prefix))
        os.makedirs(d, exist_ok=True)
        free = shutil.disk_usage(d).free
        if need * 1.02 > free:
            raise OSError(f'cache needs {need / 1e9:.1f} GB but only {free / 1e9:.1f} GB free in {d}')
        p = _paths(prefix)
        np.memmap(p['images'], np.uint8, 'w+', shape=(N, S, D, H, H)).flush()
        np.memmap(p['valid'], np.uint8, 'w+', shape=(N, S, D)).flush()
        np.memmap(p['slot'], np.uint8, 'w+', shape=(N, S)).flush()
        np.memmap(p['done'], np.uint8, 'w+', shape=(N,)).flush()
        with open(p['meta'], 'w') as fh:
            json.dump(dict(studies=list(studies), N=N, S=S, D=D, H=H, cfg=asdict(cfg)), fh)
        return StudyCache(prefix, 'r+')

    def slot_stack(self, i, s):
        return self.images[i, s]

    def flush(self):
        for a in (self.images, self.valid, self.slot, self.done):
            a.flush()


def cfg_of(prefix):
    """The PreCfg a cache was built with (single source of truth for training/inference on that cache)."""
    with open(_paths(prefix)['meta']) as fh:
        return config.cfg_from_dict(json.load(fh)['cfg'])


def _work(i):
    g = _G
    st = g['studies'][i]
    res = pipeline.prepare_study(st, g['slot_rows'].get(st, {}), g['records'], g['sides'].get(st, {}).get('side', 'U'), g['cfg'])
    c = g['cache']
    c.images[i] = res['stack']
    c.valid[i] = res['valid']
    c.slot[i] = res['slot_mask']
    c.done[i] = 1
    errs = sum(len(v.get('errors', [])) for v in res['info'].values())
    pad = sum(bool(v.get('padded')) for v in res['info'].values())
    miss = sum(bool(v.get('ps_missing')) for v in res['info'].values())
    return i, int(res['slot_mask'].sum()), errs, pad, miss


def _cost(study, slot_rows, records):
    c = 0
    for uid in slot_rows.get(study, {}).values():
        r = records.get(uid) if uid else None
        if r:
            c += (r.get('n_slices') or 0) * (5 if str(r.get('MRAcquisitionType')) == '3D' else 1)
    return c


# FIX A: Initialize globals for ProcessPoolExecutor under "spawn" context
def _init_worker(studies, slot_rows, records, sides, cfg, prefix):
    global _G
    _G.update(studies=studies, slot_rows=slot_rows, records=records, sides=sides, cfg=cfg, cache=StudyCache(prefix, 'r+'))


def build_cache(prefix, studies, slot_rows, records, sides, cfg, workers=None, resume=True, order='lpt',
                chunk=64, on_ready=None, progress=True):
    """Build (or resume) the cache. Returns the open StudyCache (mode r+) and a stats dict.

    studies: ordered list of StudyInstanceUID. slot_rows: {study: {slot: series uid|None}}.
    on_ready(cache, a, b): called in the main thread whenever studies [a,b) are all done (use order='seq')."""
    workers = workers or max(2, os.cpu_count() or 2)
    p = _paths(prefix)
    if resume and os.path.exists(p['meta']):
        cache = StudyCache(prefix, 'r+')
        if cache.studies != list(studies) or cache.meta['cfg'] != json.loads(json.dumps(asdict(cfg))):
            raise ValueError('existing cache was built for different studies/cfg; use resume=False or another prefix')
    else:
        cache = StudyCache.create(prefix, studies, cfg)
    todo = [i for i in range(len(studies)) if not cache.done[i]]
    if order == 'lpt':
        todo.sort(key=lambda i: -_cost(studies[i], slot_rows, records))
    _G.update(studies=list(studies), slot_rows=slot_rows, records=records, sides=sides, cfg=cfg, cache=cache)
    try:
        ex = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('fork'))
    except Exception:
        # FIX A: Use spawn instead of GIL-blocked ThreadPoolExecutor
        ex = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context('spawn'),
            initializer=_init_worker,
            initargs=(list(studies), slot_rows, records, sides, cfg, prefix)
        )
    t0 = time.time()
    stats = dict(studies=len(studies), built=0, resumed=len(studies) - len(todo), decode_errors=0, padded=0, ps_missing=0,
                 empty_studies=0)
    futs = {ex.submit(_work, i): i for i in todo}
    pending = {}
    for a in range(0, len(studies), chunk):
        pending[a] = sum(1 for i in range(a, min(a + chunk, len(studies))) if not cache.done[i])
    for a, left in list(pending.items()):
        if left == 0 and on_ready:
            on_ready(cache, a, min(a + chunk, len(studies)))
        if left == 0:
            pending.pop(a)
    for k, f in enumerate(as_completed(futs)):
        i, nslot, errs, pad, miss = f.result()
        stats['built'] += 1
        stats['decode_errors'] += errs
        stats['padded'] += pad
        stats['ps_missing'] += miss
        stats['empty_studies'] += int(nslot == 0)
        a = (i // chunk) * chunk
        if a in pending:
            pending[a] -= 1
            if pending[a] == 0:
                if on_ready:
                    on_ready(cache, a, min(a + chunk, len(studies)))
                pending.pop(a)
        if progress and (k + 1) % 200 == 0:
            elapsed = time.time() - t0
            throughput = (k + 1) / elapsed
            left = len(todo) - (k + 1)
            eta = left / throughput if throughput > 0 else 0
            eta_m, eta_s = divmod(int(eta), 60)
            eta_h, eta_m = divmod(eta_m, 60)
            print(f'  cache {k + 1}/{len(todo)} | {throughput:.1f} studies/s | ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}', flush=True)
    ex.shutdown()
    cache.flush()
    stats['seconds'] = time.time() - t0
    pd.DataFrame(dict(StudyInstanceUID=studies, side=[sides.get(s, {}).get('side', 'U') for s in studies],
                      side_source=[sides.get(s, {}).get('source', 'none') for s in studies],
                      n_slots=np.asarray(cache.slot).sum(1))).to_csv(p['csv'], index=False)
    # MEMORY AUDIT FIX: Purge module-level global dict to release all records and series metadata
    _G.clear()
    import gc
    gc.collect()
    return cache, stats
