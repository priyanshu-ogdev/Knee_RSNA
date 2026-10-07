"""Library-level entry points used by the notebook: index -> slots/laterality -> cache -> QC.

Everything is checkpointed/resumable; every step prints the numbers that serve as acceptance gates."""
import os
import json
import hashlib
import time
import shutil
import tempfile

import numpy as np
import pandas as pd

import src.core.config as config
from . import index as pix
from . import slots as pslots
from . import pipeline
from . import cache as pcache
from . import qc


# ----------------------------------------------------------------------------- environment
def pick_scratch(candidates=None, verbose=True):
    """Largest writable scratch disk OUTSIDE /kaggle/working (which is capped at ~20 GB and persisted as output)."""
    cands = candidates or ['/kaggle/temp', '/kaggle/tmp', '/tmp', tempfile.gettempdir()]
    best, table = None, []
    for c in dict.fromkeys(cands):
        try:
            if not os.path.isdir(c) or not os.access(c, os.W_OK):      # only existing, writable dirs (no side effects)
                continue
            free = shutil.disk_usage(c).free / 1e9
            table.append((c, round(free, 1)))
            if best is None or free > best[1]:
                best = (c, free)
        except OSError:
            continue
    if verbose:
        print('scratch candidates (path, free GB):', table)
    if best is None:
        raise OSError('no writable scratch directory found')
    return best[0]


def estimate_cache_gb(n_studies, cfg):
    return n_studies * config.N_SLOTS * cfg.stack_depth * cfg.img_size ** 2 / 1e9


def fit_cache_cfg(n_studies, free_gb, preset='v2', margin=0.92):
    """Largest standard (img_size, stack_depth) that fits the free disk. The 'public' preset is fixed (D=12, 336 px)."""
    if preset == 'public':
        return config.get_cfg('public')
    ladder = [(336, 24), (336, 20), (320, 20), (288, 24), (288, 20), (256, 20), (256, 16), (224, 16)]
    for img, depth in ladder:
        cfg = config.get_cfg(preset, img_size=img, stack_depth=depth)
        if estimate_cache_gb(n_studies, cfg) <= free_gb * margin:
            return cfg
    return config.get_cfg(preset, img_size=224, stack_depth=12)


# ----------------------------------------------------------------------------- 1. index
def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, payload):
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=1, sort_keys=True, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_index(root, out_dir, splits=('train', 'test'), workers=None, limit=0, preset='v2', progress=True, force=False):
    """Scan every series on disk, join CSV flags, assign slots, resolve laterality. Saves to out_dir:
    index.pkl, slots.pkl, sides.json, series_table.csv.gz, report.json. -> dict(ann, tab, sides, report)."""
    cfg = config.get_cfg(preset)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    df = pix.build_index(
        root, tuple(splits), workers, 1000, os.path.join(out_dir, 'chunks'), limit,
        force=force, progress=progress,
    )
    dicom_signature = df.attrs.get("source_signature")
    df = pix.attach_csv_flags(df, root)
    relevant_csvs = [
        os.path.join(root, name)
        for name in ("train_series.csv", "test_series.csv")
        if os.path.exists(os.path.join(root, name))
        and (name.startswith("train") or "test" in splits)
    ]
    csv_signatures = {os.path.basename(path): _sha256_file(path) for path in relevant_csvs}
    source_signature = hashlib.sha256(
        json.dumps(
            {
                "dicom_signature": dicom_signature,
                "csv_signatures": csv_signatures,
                "preprocess_schema": "index-slots-v2",
                "preset": cfg.name,
                "slot_prefer_2d": cfg.slot_prefer_2d,
                "slot_fs_priority": cfg.slot_fs_priority,
                "slot_csv_fallback": cfg.slot_csv_fallback,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    df.attrs["source_signature"] = source_signature
    rep = pix.index_report(df)
    rep['scan_minutes'] = round((time.time() - t0) / 60, 2)
    rep['source_signature'] = source_signature
    rep['csv_signatures'] = csv_signatures
    ann, tab = pslots.assign_all(df, cfg.slot_prefer_2d, cfg.slot_fs_priority, cfg.slot_csv_fallback)
    sides = pipeline.study_sides(df)
    rep['slot_coverage_pct'] = (tab.notna().mean() * 100).round(1).to_dict()
    rep['slots_filled_per_study'] = {int(k): int(v) for k, v in tab.notna().sum(axis=1).value_counts().sort_index().items()}
    rep['laterality'] = qc.laterality_report(sides)
    rep['header_vs_csv_fatsat_conflicts'] = int(ann['fs_conflict'].sum()) if 'fs_conflict' in ann else 0
    pix.save_index(ann, os.path.join(out_dir, 'index.pkl'))
    slots_path = os.path.join(out_dir, 'slots.pkl')
    pd.to_pickle(tab, f"{slots_path}.tmp")
    os.replace(f"{slots_path}.tmp", slots_path)
    _atomic_json(os.path.join(out_dir, 'sides.json'), sides)
    ann.drop(columns=['ordered_files', 'positions'], errors='ignore').to_csv(
        os.path.join(out_dir, 'series_table.csv.gz.tmp'), index=False, compression="gzip"
    )
    os.replace(
        os.path.join(out_dir, 'series_table.csv.gz.tmp'),
        os.path.join(out_dir, 'series_table.csv.gz'),
    )
    _atomic_json(os.path.join(out_dir, 'report.json'), rep)
    _atomic_json(
        os.path.join(out_dir, 'index_manifest.json'),
        {
            "schema_version": "index-v2",
            "source_signature": source_signature,
            "dicom_signature": dicom_signature,
            "csv_signatures": csv_signatures,
            "series": len(ann),
            "studies": int(ann['StudyInstanceUID'].nunique()) if not ann.empty else 0,
        },
    )
    lat = rep['laterality']
    print(json.dumps(rep, indent=1, default=str))
    print('\nGATES (index):')
    print(f"  series with read errors          : {rep['series_with_error']}  (expect ~0)")
    print(f"  non-canonical orientation series : {rep.get('non_canonical_orientation')}  (expect 0)")
    print(f"  CSV plane vs DICOM geometry      : {rep.get('csv_plane_vs_geometry_mismatch')} mismatches (expect 0)")
    print(f"  laterality unresolved            : {lat['unresolved_frac']:.2%}  (gate < 1%)")
    print(f"  laterality tag-vs-geometry clash : {lat['tag_geometry_disagree']} studies  (gate < 1% of tagged)")
    return dict(ann=ann, tab=tab, sides=sides, report=rep, source_signature=source_signature)


def get_index(root, out_dir, splits=('train', 'test'), workers=None, force=False, **kw):
    """Revalidate current DICOM/CSV fingerprints, then reuse only matching index chunks."""
    return run_index(root, out_dir, splits, workers, force=force, **kw)


def load_index_dir(out_dir):
    return (pix.load_index(os.path.join(out_dir, 'index.pkl')), pd.read_pickle(os.path.join(out_dir, 'slots.pkl')),
            json.load(open(os.path.join(out_dir, 'sides.json'))))


# ----------------------------------------------------------------------------- 2. cache
def run_cache(index, split, prefix, preset='v2', workers=None, limit_studies=0, fresh=False, cfg=None, studies=None,
              **cfg_overrides):
    """Build/resume the memmap cache for a split. `index` is the run_index() dict or an index directory.
    -> (cache, stats)."""
    if isinstance(index, dict):
        ann, tab, sides = index['ann'], index['tab'], index['sides']
        source_signature = index.get("source_signature")
    else:
        ann, tab, sides = load_index_dir(index)
        manifest_path = os.path.join(index, "index_manifest.json")
        if not os.path.exists(manifest_path):
            raise ValueError(f"Missing index provenance manifest: {manifest_path}; rebuild the index")
        with open(manifest_path, encoding="utf-8") as stream:
            source_signature = json.load(stream).get("source_signature")
        if not source_signature:
            raise ValueError(f"Invalid index provenance manifest: {manifest_path}")
    cfg = cfg or config.get_cfg(preset, **cfg_overrides)
    if studies is None:
        studies = sorted(ann[ann['split'] == split]['StudyInstanceUID'].unique())
    if limit_studies:
        studies = studies[:limit_studies]
    need = estimate_cache_gb(len(studies), cfg)
    print(f'{len(studies)} studies -> cache {need:.1f} GB  (preset={cfg.name}, D={cfg.stack_depth}, {cfg.img_size}px) at {prefix}')
    sub = ann[ann['StudyInstanceUID'].isin(set(studies))] if ('StudyInstanceUID' in ann.columns) else ann
    records = pipeline.index_to_records(sub)
    slot_rows = {
        s: {
            name: (
                str(value).strip()
                if pd.notna(value) and str(value).strip()
                else None
            )
            for name, value in tab.loc[s].to_dict().items()
        }
        for s in studies
        if s in tab.index
    }
    cache, stats = pcache.build_cache(
        prefix,
        studies,
        slot_rows,
        records,
        sides,
        cfg,
        workers,
        resume=not fresh,
        source_signature=source_signature,
    )
    print(json.dumps(stats, indent=1))
    return cache, stats


# ----------------------------------------------------------------------------- 3. QC
def run_qc(prefix, out_dir=None, n=300, montage=True, expected_slot_mask=None, require_acceptance=True):
    """Acceptance gates on a built cache (+ optional montage PNG: one row per slot type)."""
    c = pcache.StudyCache(prefix)
    rep = qc.cache_sanity(c, n, expected_slot_mask=expected_slot_mask)
    rep["source_signature"] = c.meta.get("source_signature")
    print(json.dumps(rep, indent=1, default=str))
    print(f"\nGATES (cache): accepted={rep.get('accepted')} "
          f"incomplete={rep.get('incomplete_studies')} decode_errors={rep.get('decode_errors')} "
          f"failed_selected_slots={rep.get('failed_selected_slots')} "
          f"missing_expected_slots={rep.get('missing_expected_slots')} "
          f"mean_valid_depth={rep.get('mean_valid_depth', float('nan')):.2f}")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        _atomic_json(os.path.join(out_dir, 'cache_sanity.json'), rep)
    if montage and out_dir:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            rng = np.random.default_rng(0)
            done = np.where(np.asarray(c.done) == 1)[0]
            fig, ax = plt.subplots(config.N_SLOTS, 6, figsize=(14, 2.4 * config.N_SLOTS))
            for s in range(config.N_SLOTS):
                cand = [i for i in done if c.slot[i, s]]
                pick = rng.choice(cand, min(6, len(cand)), replace=False) if cand else []
                for j in range(6):
                    ax[s, j].axis('off')
                for j, i in enumerate(pick):
                    ax[s, j].imshow(c.images[i, s, c.images.shape[2] // 2], cmap='gray', vmin=0, vmax=255)
                    ax[s, j].set_title(config.SLOTS[s][0], fontsize=7)
            plt.tight_layout()
            path = os.path.join(out_dir, 'montage.png')
            plt.savefig(path, dpi=90)
            plt.close('all')
            print('montage ->', path)
        except Exception as e:                                 # a plotting problem must never fail the pipeline
            print('montage skipped:', type(e).__name__, e)
    if require_acceptance and not rep.get("accepted", False):
        raise RuntimeError(
            f"Preprocessing cache failed acceptance gates; see "
            f"{os.path.join(out_dir, 'cache_sanity.json') if out_dir else prefix + '.meta.csv'}"
        )
    return rep
