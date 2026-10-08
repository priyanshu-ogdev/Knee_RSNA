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
                                                 done='done.u1', errors='errors.u4', failed='failed.u1',
                                                 padded='padded.u1', ps_missing='ps_missing.u1',
                                                 meta='meta.json', csv='meta.csv').items()}


class StudyCache:
    def __init__(self, prefix, mode='r'):
        self.prefix, self.mode = prefix, mode
        p = _paths(prefix)
        with open(p['meta']) as fh:
            self.meta = json.load(fh)
        if self.meta.get("schema_version") != "preprocess-cache-v2":
            raise ValueError(
                f"Unsupported or legacy cache metadata at {p['meta']}; rebuild with --fresh_preprocessing"
            )
        self.studies = self.meta['studies']
        N, S, D, H = self.meta['N'], self.meta['S'], self.meta['D'], self.meta['H']
        expected_sizes = {
            'images': N * S * D * H * H,
            'valid': N * S * D,
            'slot': N * S,
            'done': N,
            'errors': N * 4,
            'failed': N * S,
            'padded': N * S,
            'ps_missing': N * S,
        }
        for key, expected_size in expected_sizes.items():
            actual_size = os.path.getsize(p[key]) if os.path.exists(p[key]) else -1
            if actual_size != expected_size:
                raise ValueError(
                    f"Cache storage mismatch for {p[key]}: expected {expected_size} bytes, "
                    f"found {actual_size}; rebuild with --fresh_preprocessing"
                )
        m = 'r+' if mode == 'r+' else 'r'
        self.images = np.memmap(p['images'], np.uint8, m, shape=(N, S, D, H, H))
        self.valid = np.memmap(p['valid'], np.uint8, m, shape=(N, S, D))
        self.slot = np.memmap(p['slot'], np.uint8, m, shape=(N, S))
        self.done = np.memmap(p['done'], np.uint8, m, shape=(N,))
        self.errors = np.memmap(p['errors'], np.uint32, m, shape=(N,))
        self.failed = np.memmap(p['failed'], np.uint8, m, shape=(N, S))
        self.padded = np.memmap(p['padded'], np.uint8, m, shape=(N, S))
        self.ps_missing = np.memmap(p['ps_missing'], np.uint8, m, shape=(N, S))
        self.index = {s: i for i, s in enumerate(self.studies)}
        # Hint random-access memmaps to avoid excessive read-ahead; this is advisory.
        try:
            import mmap as _py_mmap
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
    def create(prefix, studies, cfg, source_signature=None):
        studies = list(studies)
        if not studies or len(studies) != len(set(studies)):
            raise ValueError("Cache studies must be a non-empty list of unique StudyInstanceUID values")
        N, S, D, H = len(studies), config.N_SLOTS, cfg.stack_depth, cfg.img_size
        need = N * S * D * H * H + N * S * D + N * S + N + 4 * N + 3 * N * S
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
        np.memmap(p['errors'], np.uint32, 'w+', shape=(N,)).flush()
        np.memmap(p['failed'], np.uint8, 'w+', shape=(N, S)).flush()
        np.memmap(p['padded'], np.uint8, 'w+', shape=(N, S)).flush()
        np.memmap(p['ps_missing'], np.uint8, 'w+', shape=(N, S)).flush()
        meta = dict(
            schema_version="preprocess-cache-v2",
            studies=studies,
            N=N,
            S=S,
            D=D,
            H=H,
            cfg=asdict(cfg),
            source_signature=source_signature,
        )
        temporary = f"{p['meta']}.tmp"
        with open(temporary, 'w', encoding="utf-8") as fh:
            json.dump(meta, fh, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temporary, p['meta'])
        return StudyCache(prefix, 'r+')

    def slot_stack(self, i, s):
        return self.images[i, s]

    def flush(self):
        for a in (
            self.images, self.valid, self.slot, self.done, self.errors,
            self.failed, self.padded, self.ps_missing,
        ):
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
    expected = np.array(
        [bool(g['slot_rows'].get(st, {}).get(name)) for name, _, _, _ in config.SLOTS],
        dtype=bool,
    )
    slot_errors = np.array(
        [bool(res['info'].get(name, {}).get('errors')) for name, _, _, _ in config.SLOTS],
        dtype=bool,
    )
    failed = expected & ((res['slot_mask'] == 0) | slot_errors)
    errs = sum(len(v.get('errors', [])) for v in res['info'].values())
    c.errors[i] = errs
    c.failed[i] = failed.astype(np.uint8)
    c.padded[i] = np.array(
        [bool(res['info'].get(name, {}).get('padded')) for name, _, _, _ in config.SLOTS],
        dtype=np.uint8,
    )
    c.ps_missing[i] = np.array(
        [bool(res['info'].get(name, {}).get('ps_missing')) for name, _, _, _ in config.SLOTS],
        dtype=np.uint8,
    )
    c.done[i] = 1
    pad = sum(bool(v.get('padded')) for v in res['info'].values())
    miss = sum(bool(v.get('ps_missing')) for v in res['info'].values())
    return i, int(res['slot_mask'].sum()), errs, pad, miss, int(failed.sum())


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
                chunk=64, on_ready=None, progress=True, source_signature=None):
    """Build (or resume) the cache. Returns the open StudyCache (mode r+) and a stats dict.

    studies: ordered list of StudyInstanceUID. slot_rows: {study: {slot: series uid|None}}.
    on_ready(cache, a, b): called in the main thread whenever studies [a,b) are all done (use order='seq')."""
    workers = workers or max(2, os.cpu_count() or 2)
    p = _paths(prefix)
    if source_signature is None:
        import hashlib
        import json as _json
        selection = {
            study: slot_rows.get(study, {})
            for study in studies
        }
        source_signature = hashlib.sha256(
            _json.dumps(selection, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
    if resume and os.path.exists(p['meta']):
        cache = StudyCache(prefix, 'r+')
        if (
            cache.studies != list(studies)
            or cache.meta['cfg'] != json.loads(json.dumps(asdict(cfg)))
            or cache.meta.get("source_signature") != source_signature
        ):
            raise ValueError(
                "existing cache was built from different studies, slots, DICOMs, or preprocessing config; "
                "use --fresh_preprocessing or another cache prefix"
            )
    else:
        cache = StudyCache.create(prefix, studies, cfg, source_signature)
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
    stats = dict(studies=len(studies), built=0, resumed=len(studies) - len(todo))
    try:
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
            i, _, _, _, _, _ = f.result()
            stats['built'] += 1
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
                print(
                    f'  cache {k + 1}/{len(todo)} | {throughput:.1f} studies/s | '
                    f'ETA: {eta_h:02d}:{eta_m:02d}:{eta_s:02d}',
                    flush=True,
                )
    except Exception:
        ex.shutdown(wait=True)
        cache.flush()
        _G.clear()
        raise
    else:
        ex.shutdown(wait=True)
    cache.flush()
    stats.update(
        decode_errors=int(np.asarray(cache.errors).sum()),
        failed_selected_slots=int(np.asarray(cache.failed).sum()),
        padded=int(np.asarray(cache.padded).sum()),
        ps_missing=int(np.asarray(cache.ps_missing).sum()),
        empty_studies=int((np.asarray(cache.slot).sum(axis=1) == 0).sum()),
        incomplete_studies=int((np.asarray(cache.done) != 1).sum()),
        source_signature=source_signature,
    )
    stats['seconds'] = time.time() - t0
    meta_frame = pd.DataFrame(dict(
        StudyInstanceUID=studies,
        side=[sides.get(s, {}).get('side', 'U') for s in studies],
        side_source=[sides.get(s, {}).get('source', 'none') for s in studies],
        n_slots=np.asarray(cache.slot).sum(1),
        decode_errors=np.asarray(cache.errors),
        failed_selected_slots=np.asarray(cache.failed).sum(1),
        padded=np.asarray(cache.padded).sum(1),
        ps_missing=np.asarray(cache.ps_missing).sum(1),
    ))
    meta_tmp = f"{p['csv']}.tmp"
    meta_frame.to_csv(meta_tmp, index=False)
    os.replace(meta_tmp, p['csv'])
    # MEMORY AUDIT FIX: Purge module-level global dict to release all records and series metadata
    _G.clear()
    import gc
    gc.collect()
    return cache, stats
