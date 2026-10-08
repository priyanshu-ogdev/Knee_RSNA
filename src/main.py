"""
RSNA 2026: End-to-End Master Training & Build Pipeline
------------------------------------------------------
Executes all phases seamlessly in a unified workflow:
  PHASE 0: Environment & Hardware Configuration + Dataset Verification
  PHASE 1: NLP Pseudo-Label Auto-Detection & Completion (vLLM / Clinical Rules)
  PHASE 2: Dataset Merging, Stratification & Cache Build (Download-Aware)
  PHASE 3: 5-Fold Deep Learning Training (DINOv2 or CoAtNet attention-MIL)
  PHASE 4: Out-Of-Fold Evaluation & Checkpoint Verification
  PHASE 5: Test Inference & Submission Generation (TTA; identity temperature)
  PHASE 6: Final Telemetry & Execution Summary
"""
from __future__ import annotations

import os
import sys
import shutil
import time
import json
import math
import argparse
import datetime
import traceback
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Ensure stdout handles UTF-8 characters cleanly
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Project root resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Force kagglehub to download directly into root data/ directory
os.environ["KAGGLEHUB_CACHE"] = os.path.abspath(os.path.join(PROJECT_ROOT, "data"))

# PyTorch Memory Allocation: prevent unified memory fragmentation on Grace Blackwell GB10
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Suppress OpenCV / OpenMP / BLAS thread oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    import cv2
    cv2.setNumThreads(0)
except Exception:
    pass

# Load environment variables from .env
load_dotenv()

import torch
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

import src.core.config as config
from src.data.labels import build_labels
from src.data.preprocess import runner, splits
from src.training.train import run_training, MemoryCircuitBreakerTriggered, execute_emergency_memory_flush
from src.inference.inference import run_inference
from src.modeling.model import load_checkpoint


# ==============================================================================
# PHASE 0: HARDWARE & ENVIRONMENT VERIFICATION
# ==============================================================================
def verify_hardware_and_environment():
    print("=" * 80)
    print("PHASE 0: HARDWARE & ENVIRONMENT VERIFICATION")
    print("=" * 80)
    
    cuda_avail = torch.cuda.is_available()
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {cuda_avail}")
    if cuda_avail:
        dev_count = torch.cuda.device_count()
        dev_name = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        bf16_sup = torch.cuda.is_bf16_supported()
        print(f"Device Count      : {dev_count}")
        print(f"Primary GPU       : {dev_name} (Compute Capability: {capability[0]}.{capability[1]})")
        print(f"GPU VRAM Total    : {vram_gb:.2f} GB")
        print(f"BFloat16 Native   : {bf16_sup}")
        print(f"CUDA Alloc Config : {os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}")
    else:
        print("[WARNING] CUDA is NOT available. Pipeline will run on CPU.")

    cpu_cores = os.cpu_count() or 4
    print(f"System CPU Cores  : {cpu_cores}")
    print(f"Platform OS       : {sys.platform}")


def resolve_data_root(cli_data_root: str | None = None) -> str:
    if cli_data_root:
        resolved = config.resolve_data_root(cli_data_root)
        if os.path.exists(os.path.join(resolved, "train.csv")):
            print(f"[SUCCESS] Dataset located via CLI argument: {resolved}")
            return os.path.abspath(resolved)

    knee_env = os.environ.get("KNEE_DATA")
    if knee_env and os.path.exists(os.path.join(knee_env, "train.csv")):
        data_root = os.path.abspath(knee_env)
        print(f"[SUCCESS] Dataset located via KNEE_DATA: {data_root}")
        return data_root

    local_data = os.path.abspath(os.path.join(PROJECT_ROOT, "data"))
    if os.path.exists(os.path.join(local_data, "train.csv")):
        print(f"[SUCCESS] Dataset located in local directory: {local_data}")
        return local_data

    print("[INFO] train.csv not found locally. Checking/Downloading via Kagglehub...")
    try:
        import kagglehub
        path = kagglehub.competition_download("rsna-knee-abnormality-detection")
        print(f"[SUCCESS] Dataset downloaded via kagglehub to: {path}")
        return os.path.abspath(path)
    except Exception as e:
        print(f"[WARNING] kagglehub download failed: {e}")
        print(f"[FALLBACK] Falling back to default data path: {local_data}")
        return local_data


# ==============================================================================
# PHASE 1: NLP PSEUDO-LABEL AUTO-DETECTION & COMPLETION
# ==============================================================================
def run_nlp_phase(
    data_root: str,
    work_dir: str,
    skip_nlp: bool = False,
    engine: str = "auto",
    model_id: str | None = None,
    force: bool = False,
) -> str | None:
    print("\n" + "=" * 80)
    print("PHASE 1: NLP PSEUDO-LABEL AUTO-DETECTION & COMPLETION")
    print("=" * 80)

    if skip_nlp:
        print("[INFO] --skip_nlp specified. Skipping extraction and using labels present in train.csv only.")
        return None

    # Keep generated labels isolated to this run directory. External CSVs are
    # never consumed without the extractor's matching provenance manifest.
    out_csv = os.path.join(work_dir, "pseudo_labels.csv")

    from src.data.preprocess.nlp_extractor import auto_complete_extraction
    pseudo_csv, stats = auto_complete_extraction(
        data_root=data_root,
        out_csv=out_csv,
        model_id=model_id,
        engine=engine,
        force=force,
    )
    print(
        f"[SUCCESS] NLP labels verified: {stats['total']} studies "
        f"(new={stats['new']}, engine={stats['engine']})"
    )
    return pseudo_csv


# ==============================================================================
# PHASE 2: DATASET MERGING, STRATIFICATION & CACHE BUILD
# ==============================================================================
def _sha256_file(path: str) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: str, payload: dict) -> None:
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, default=str)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_folds(folds_df: pd.DataFrame, folds_csv: str) -> None:
    temporary = f"{folds_csv}.tmp"
    folds_df.to_csv(temporary, index=False)
    os.replace(temporary, folds_csv)


def _fold_contract(folds_df: pd.DataFrame, study_uids: set[str], duplicate_pairs) -> None:
    if folds_df["StudyInstanceUID"].duplicated().any():
        raise ValueError("folds.csv contains duplicate StudyInstanceUID values")
    fold_uids = set(folds_df["StudyInstanceUID"].astype(str))
    if fold_uids != study_uids:
        missing = sorted(study_uids - fold_uids)
        extra = sorted(fold_uids - study_uids)
        raise ValueError(
            f"folds.csv does not cover exactly the training studies "
            f"(missing={missing[:5]}, extra={extra[:5]})"
        )
    folds = pd.to_numeric(folds_df["fold"], errors="coerce")
    if (
        folds.isna().any()
        or not np.equal(folds, folds.astype(int)).all()
        or set(folds.astype(int)) != set(range(5))
    ):
        raise ValueError("folds.csv must assign studies to all five folds (0..4)")
    assigned = folds_df.assign(fold=folds.astype(int)).set_index("StudyInstanceUID")["fold"]
    violations = [
        (a, b)
        for a, b in duplicate_pairs
        if a in study_uids and b in study_uids and assigned[a] != assigned[b]
    ]
    if violations:
        raise ValueError(f"Verified duplicate exams were split across folds: {violations[:5]}")


def run_preparation(
    data_root: str,
    work_dir: str,
    pseudo_csv: str | None,
    force: bool = False,
    workers: int | None = None,
) -> tuple[str, str | None, str]:
    print("\n" + "=" * 80)
    print("PHASE 2: DATASET MERGE, STRATIFICATION & CACHE BUILD")
    print("=" * 80)

    final_labels_csv = os.path.join(work_dir, "train_labels_v2.csv")
    print(f"Merging Gold labels (weight 1.0) and pseudo-labels (weight 0.5) -> {final_labels_csv}...")
    labels_df = build_labels(data_root, extra_csv=pseudo_csv, extra_weight=0.5, out_csv=final_labels_csv)
    target_coverage = {
        target: {
            "labeled": int(labels_df[target].notna().sum()),
            "weighted": int((labels_df[f"{target}_weight"] > 0).sum()),
            "positive": int((labels_df[target] > 0).sum()),
        }
        for target in config.TARGETS
    }
    print(
        f"[SUCCESS] Labels assembled: {len(labels_df)} studies; "
        f"source counts={labels_df['source'].value_counts().to_dict()}"
    )
    train_csv = os.path.join(data_root, "train.csv")
    train_series_csv = os.path.join(data_root, "train_series.csv")
    train_raw = pd.read_csv(train_csv)
    if "Report" not in train_raw.columns:
        raise ValueError("train.csv must include the radiology Report column for NLP label generation")
    if not os.path.isfile(train_series_csv):
        raise FileNotFoundError(f"Required competition series index is missing: {train_series_csv}")
    series_csv = pd.read_csv(train_series_csv)
    required_series_columns = {
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Anatomical_Plane",
        "Fluid_Sensitive",
        "Fat_Suppression",
    }
    if not required_series_columns.issubset(series_csv.columns):
        raise ValueError(f"train_series.csv must contain {sorted(required_series_columns)}")
    if series_csv[list(required_series_columns)].isna().any().any():
        raise ValueError("train_series.csv contains missing identifiers or acquisition flags")
    series_csv["StudyInstanceUID"] = series_csv["StudyInstanceUID"].astype(str).str.strip()
    series_csv["SeriesInstanceUID"] = series_csv["SeriesInstanceUID"].astype(str).str.strip()
    if series_csv["SeriesInstanceUID"].eq("").any() or series_csv["SeriesInstanceUID"].duplicated().any():
        raise ValueError("train_series.csv must have non-empty, unique SeriesInstanceUID values")
    if not set(series_csv["Anatomical_Plane"].astype(str)).issubset({"Axial", "Coronal", "Sagittal"}):
        raise ValueError("train_series.csv contains an unknown Anatomical_Plane value")
    for flag in ("Fluid_Sensitive", "Fat_Suppression"):
        if not series_csv[flag].isin([0, 1, False, True]).all():
            raise ValueError(f"train_series.csv {flag} values must be binary")

    train_studies = set(labels_df["StudyInstanceUID"].astype(str))
    folds_csv = os.path.join(work_dir, "folds.csv")
    labels_manifest = f"{final_labels_csv}.manifest.json"
    with open(labels_manifest, encoding="utf-8") as stream:
        label_provenance = json.load(stream)
    nlp_provenance = None
    if pseudo_csv:
        nlp_manifest = f"{pseudo_csv}.manifest.json"
        if not os.path.isfile(nlp_manifest):
            raise ValueError(f"Pseudo-label file has no provenance manifest: {nlp_manifest}")
        with open(nlp_manifest, encoding="utf-8") as stream:
            nlp_provenance = json.load(stream)
        if nlp_provenance.get("status") != "complete":
            raise ValueError(f"NLP extraction is incomplete: {nlp_manifest}")
        if nlp_provenance.get("train_csv_sha256") != _sha256_file(train_csv):
            raise ValueError("NLP labels were generated from a different train.csv")
        if nlp_provenance.get("pseudo_csv_sha256") != _sha256_file(pseudo_csv):
            raise ValueError("NLP pseudo-label CSV does not match its provenance manifest")
        if int(nlp_provenance.get("completed_studies", -1)) != int(nlp_provenance.get("required_studies", -2)):
            raise ValueError("NLP provenance reports an incomplete eligible-study set")
        if int(nlp_provenance.get("completed_studies", -1)) != len(pd.read_csv(pseudo_csv)):
            raise ValueError("Pseudo-label manifest row count does not match pseudo-label CSV")

    preparation_manifest = os.path.join(work_dir, "dataset_preparation_manifest.json")
    folds_manifest_path = os.path.join(work_dir, "folds_manifest.json")
    initial_manifest = {
        "schema_version": "dataset-preparation-v2",
        "status": "in_progress",
        "data_root": os.path.abspath(data_root),
        "train_csv_sha256": _sha256_file(train_csv),
        "train_series_csv_sha256": _sha256_file(train_series_csv),
        "labels_csv_sha256": label_provenance["labels_csv_sha256"],
        "nlp": nlp_provenance,
        "target_coverage": target_coverage,
        "cache_prefix": None,
        "cache_cfg": None,
    }
    _atomic_json(preparation_manifest, initial_manifest)

    duplicate_path = os.path.join(PROJECT_ROOT, "eda", "series_meta.csv.gz")
    duplicate_pairs, duplicate_report = splits.duplicate_pairs_from_series_meta(
        duplicate_path, study_uids=train_studies, min_shared_hashes=2
    )
    print(f"[DUPLICATE HASH AUDIT] {duplicate_report}")

    if not runner.pix.list_series_dirs(data_root, "train"):
        study_meta = splits.make_study_meta(None, train_csv=train_raw, labels_df=labels_df)
        folds_df = splits.group_folds(
            study_meta, n_splits=5, seed=config.SEED, scheme="site", dup_pairs=duplicate_pairs
        )
        _fold_contract(folds_df, train_studies, duplicate_pairs)
        _write_folds(folds_df, folds_csv)
        fold_report = {
            "schema_version": "folds-v2",
            "source": "report-only-staging",
            "study_count": len(folds_df),
            "fold_counts": folds_df["fold"].value_counts().sort_index().to_dict(),
            "duplicate_hash_audit": duplicate_report,
            "source_signature": _sha256_file(train_csv) + ":" + _sha256_file(train_series_csv),
        }
        _atomic_json(folds_manifest_path, fold_report)
        _atomic_json(
            preparation_manifest,
            {
                **initial_manifest,
                "status": "awaiting_dicom",
                "folds_csv_sha256": _sha256_file(folds_csv),
                "folds_manifest": fold_report,
            },
        )
        print(f"[BLOCKED] Training handoff withheld: no train DICOM series found under {data_root}")
        print(f"[INFO] Labels/folds staged; rerun after the complete DICOM tree is present. Manifest: {preparation_manifest}")
        return final_labels_csv, None, folds_csv

    # Build an input-fingerprinted index for every train series.
    idx_dir = os.path.join(work_dir, "idx")
    cpu_cores = min(
        workers if workers is not None else config.PREPROCESS_WORKERS,
        os.cpu_count() or 1,
    )
    if cpu_cores < 1:
        raise ValueError("preprocessing workers must be at least 1")
    print("Indexing and validating every training DICOM series...")
    index_out = runner.run_index(
        data_root, idx_dir, splits=("train",), workers=cpu_cores, force=force
    )
    ann = index_out["ann"]
    indexed_studies = set(ann["StudyInstanceUID"].astype(str))
    indexed_series = set(ann["SeriesInstanceUID"].astype(str))
    csv_studies = set(series_csv["StudyInstanceUID"])
    csv_series = set(series_csv["SeriesInstanceUID"])
    if indexed_studies != train_studies:
        raise ValueError(
            f"DICOM study coverage mismatch: missing={len(train_studies - indexed_studies)}, "
            f"unexpected={len(indexed_studies - train_studies)}"
        )
    if indexed_studies != csv_studies:
        raise ValueError(
            f"train_series.csv study coverage mismatch: missing={len(train_studies - csv_studies)}, "
            f"unexpected={len(csv_studies - train_studies)}"
        )
    if indexed_series != csv_series:
        raise ValueError(
            f"DICOM/CSV series mismatch: missing_on_disk={len(csv_series - indexed_series)}, "
            f"not_in_train_series_csv={len(indexed_series - csv_series)}"
        )
    slot_studies = set(index_out["tab"].index.astype(str))
    if slot_studies != train_studies:
        raise ValueError(
            f"Slot table study coverage mismatch: missing={len(train_studies - slot_studies)}, "
            f"unexpected={len(slot_studies - train_studies)}"
        )
    if ann["err"].notna().any():
        failures = ann.loc[ann["err"].notna(), ["StudyInstanceUID", "SeriesInstanceUID", "err"]].head(10)
        raise RuntimeError(
            f"DICOM metadata indexing failures ({int(ann['err'].notna().sum())} series):\n{failures}"
        )
    index_report = index_out["report"]
    if index_report.get("empty_series", 0) != 0:
        raise RuntimeError(f"Empty DICOM series gate failed: {index_report['empty_series']}")
    if index_report.get("series_with_unreadable_slices", 0) != 0:
        raise RuntimeError(
            f"Unreadable DICOM slice-header gate failed: {index_report['series_with_unreadable_slices']}"
        )
    if index_report.get("series_with_mixed_shapes", 0) != 0:
        raise RuntimeError(f"Mixed image-size gate failed: {index_report['series_with_mixed_shapes']}")
    missing_geometry = ann["plane_geo"].isna() | ann["orient_code"].isna()
    if missing_geometry.any():
        raise RuntimeError(f"DICOM orientation/plane geometry is missing for {int(missing_geometry.sum())} series")
    if index_report.get("non_canonical_orientation", 0) != 0:
        raise RuntimeError(f"Noncanonical DICOM orientation gate failed: {index_report['non_canonical_orientation']}")
    if index_report.get("csv_plane_vs_geometry_mismatch", 0) != 0:
        raise RuntimeError(f"Series CSV plane conflicts with DICOM geometry: {index_report['csv_plane_vs_geometry_mismatch']}")
    side_report = index_report.get("laterality", {})
    if side_report.get("unresolved_frac", 1.0) >= 0.01:
        raise RuntimeError(f"Laterality unresolved gate failed: {side_report.get('unresolved_frac'):.2%}")
    tagged = sum(value.get("source") == "tag" for value in index_out["sides"].values())
    clashes = side_report.get("tag_geometry_disagree", 0)
    if tagged and clashes / tagged >= 0.01:
        raise RuntimeError(f"DICOM laterality tag/geometry conflict gate failed: {clashes}/{tagged}")

    # Regenerate folds from current index/labels; never silently reuse a stale split.
    study_meta = splits.make_study_meta(ann, train_csv=train_raw, labels_df=labels_df)
    folds_df = splits.group_folds(
        study_meta, n_splits=5, seed=config.SEED, scheme="site", dup_pairs=duplicate_pairs
    )
    _fold_contract(folds_df, train_studies, duplicate_pairs)
    _write_folds(folds_df, folds_csv)
    fold_report = {
        "schema_version": "folds-v2",
        "source": "indexed-scanner-site",
        "study_count": len(folds_df),
        "fold_counts": folds_df["fold"].value_counts().sort_index().to_dict(),
        "duplicate_hash_audit": duplicate_report,
        "index_source_signature": index_out["source_signature"],
        "train_csv_sha256": _sha256_file(train_csv),
        "labels_csv_sha256": label_provenance["labels_csv_sha256"],
    }
    _atomic_json(folds_manifest_path, fold_report)

    # Fixed v2 dimensions are recorded; automatic disk-based shape changes are disabled.
    studies_to_cache = labels_df["StudyInstanceUID"].astype(str).tolist()
    cache_dir = os.environ.get("CACHE_DIR", os.path.join(work_dir, "cache"))
    os.makedirs(cache_dir, exist_ok=True)
    cfg = config.get_cfg("v2")
    img_override = os.environ.get("CACHE_IMG_SIZE")
    depth_override = os.environ.get("CACHE_STACK_DEPTH")
    if img_override or depth_override:
        img_size = int(img_override) if img_override else cfg.img_size
        stack_depth = int(depth_override) if depth_override else cfg.stack_depth
        cfg = config.get_cfg("v2", img_size=img_size, stack_depth=stack_depth)
        print(f"[CONFIG] Explicit cache override: img_size={img_size}, stack_depth={stack_depth}")
    free_gb = shutil.disk_usage(cache_dir).free / 1e9
    need_gb = runner.estimate_cache_gb(len(studies_to_cache), cfg)
    print(
        f"[CONFIG] Cache shape={cfg.img_size}x{cfg.img_size}x{cfg.stack_depth}x{config.N_SLOTS}; "
        f"image bytes estimate={need_gb:.1f} GB; free={free_gb:.1f} GB"
    )
    if need_gb * 1.02 > free_gb:
        raise OSError(
            f"Selected cache needs at least {need_gb * 1.02:.1f} GB but only {free_gb:.1f} GB is free. "
            "Mount/free sufficient NVMe space or explicitly configure CACHE_IMG_SIZE and "
            "CACHE_STACK_DEPTH; preprocessing resolution will not be silently changed."
        )

    cache_prefix = os.path.join(cache_dir, "train")
    _, cache_stats = runner.run_cache(
        index_out,
        "train",
        cache_prefix,
        cfg=cfg,
        workers=cpu_cores,
        studies=studies_to_cache,
        fresh=force,
    )
    tab = index_out["tab"]
    expected_slot_mask = np.zeros((len(studies_to_cache), config.N_SLOTS), dtype=bool)
    for row_idx, study in enumerate(studies_to_cache):
        if study in tab.index:
            slot_row = tab.loc[study]
            expected_slot_mask[row_idx] = [
                pd.notna(slot_row.get(slot_name)) and bool(str(slot_row.get(slot_name)).strip())
                for slot_name, _, _, _ in config.SLOTS
            ]
    qc_dir = os.path.join(work_dir, "qc")
    qc_report = runner.run_qc(
        cache_prefix,
        out_dir=qc_dir,
        n=min(300, len(studies_to_cache)),
        montage=False,
        expected_slot_mask=expected_slot_mask,
    )
    final_manifest = {
        **initial_manifest,
        "status": "complete",
        "index_source_signature": index_out["source_signature"],
        "index_report": index_report,
        "indexed_studies": len(indexed_studies),
        "indexed_series": len(indexed_series),
        "folds_csv_sha256": _sha256_file(folds_csv),
        "folds_manifest": fold_report,
        "duplicate_hash_audit": duplicate_report,
        "cache_prefix": os.path.abspath(cache_prefix),
        "cache_cfg": {
            "name": cfg.name,
            "img_size": cfg.img_size,
            "stack_depth": cfg.stack_depth,
            "slots": config.N_SLOTS,
            "estimated_image_bytes_gb": need_gb,
            "free_bytes_gb_before_build": free_gb,
        },
        "cache_stats": cache_stats,
        "qc_report": qc_report,
    }
    _atomic_json(preparation_manifest, final_manifest)
    print(f"[SUCCESS] Data preprocessing accepted; training handoff manifest: {preparation_manifest}")
    return final_labels_csv, cache_prefix, folds_csv


# ==============================================================================
# PHASE 3: 5-FOLD DEEP LEARNING MODEL TRAINING
# ==============================================================================
def run_all_folds(
    labels_csv: str,
    cache_prefix: str,
    folds_csv: str,
    work_dir: str,
    folds_to_run: list[int],
    epochs: int | None = None,
    batch_size: int = config.BATCH_SIZE,
    grad_accum: int = config.GRAD_ACCUM,
    n_windows_train: int = config.N_WINDOWS_TRAIN,
    num_workers: int = config.NUM_WORKERS,
    prefetch_factor: int = config.PREFETCH_FACTOR,
    eval_batch_size: int = config.EVAL_BATCH_SIZE,
    variant: str = "dinov2-base",
    model_type: str = "dinov2",
    pretrained: bool = True,
    timm_pooling: str = "hierarchical",
    use_slot_prior: bool = True,
) -> dict[int, float]:
    print("\n" + "=" * 80)
    print("PHASE 3: 5-FOLD MODEL TRAINING")
    print(f"Target Folds : {folds_to_run} | Model: {model_type}/{variant} | Batch Size: {batch_size} (accum: {grad_accum})")
    print("=" * 80)

    best_scores: dict[int, float] = {}

    if not folds_to_run or len(folds_to_run) != len(set(folds_to_run)):
        raise ValueError("folds_to_run must be a non-empty list of unique fold IDs")
    if any(fold not in range(5) for fold in folds_to_run):
        raise ValueError(f"fold IDs must be in 0..4: {folds_to_run}")
    for fold in folds_to_run:
        fold_out_dir = os.path.join(work_dir, f"models_fold{fold}")
        existing_outputs = [
            os.path.join(fold_out_dir, f"fold{fold}_{name}.pt")
            for name in ("best", "ema", "swa")
        ] + [os.path.join(fold_out_dir, f"fold{fold}_oof.csv")]
        if os.path.isdir(fold_out_dir):
            existing_outputs.extend(
                os.path.join(fold_out_dir, name)
                for name in os.listdir(fold_out_dir)
                if name.startswith(f"fold{fold}_emergency_") and name.endswith("_checkpoint.pt")
            )
        existing_outputs = [path for path in existing_outputs if os.path.exists(path)]
        if existing_outputs:
            raise FileExistsError(
                f"Fold {fold} already has artifacts; refusing to mix a new run with stale "
                f"checkpoints/OOF files. Choose a fresh --model_dir. Existing: {existing_outputs}"
            )
    for fold in folds_to_run:
        print(f"\n{'='*35} STARTING FOLD {fold} {'='*35}")
        fold_out_dir = os.path.join(work_dir, f"models_fold{fold}")
        os.makedirs(fold_out_dir, exist_ok=True)
        try:
            best_auc = run_training(
                labels_csv=labels_csv,
                cache_prefix=cache_prefix,
                folds_csv=folds_csv,
                fold=fold,
                out_dir=fold_out_dir,
                epochs=epochs,
                batch_size=batch_size,
                grad_accum=grad_accum,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                eval_batch_size=eval_batch_size,
                n_windows_train=n_windows_train,
                variant=variant,
                model_type=model_type,
                timm_pooling=timm_pooling,
                use_slot_prior=use_slot_prior,
                pretrained=pretrained,
                use_cross_slot=True,
                swa_epochs=config.SWA_EPOCHS,
            )
            best_scores[fold] = best_auc
            print(f"[SUCCESS] Fold {fold} completed with Best Macro-AUC: {best_auc:.4f}")
        except MemoryCircuitBreakerTriggered as mem_err:
            print(f"\n[CIRCUIT BREAKER] Hard memory limit reached on Fold {fold}: {mem_err.used_gb:.2f} GB >= {mem_err.threshold_gb:.2f} GB.", flush=True)
            print("[CIRCUIT BREAKER] Aborting remaining folds to protect DGX system stability.", flush=True)
            execute_emergency_memory_flush()
            raise mem_err
        except Exception as e:
            print(f"[ERROR] Error during training Fold {fold}: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Training failed for requested fold {fold}; stopping to avoid a partial ensemble") from e
        finally:
            execute_emergency_memory_flush()

    return best_scores


# ==============================================================================
# PHASE 4: OUT-OF-FOLD EVALUATION & CHECKPOINT VERIFICATION
# ==============================================================================
def run_oof_and_checkpoint_verification(
    model_dir: str,
    labels_csv: str,
    folds_csv: str,
    requested_folds: list[int],
    best_scores: dict[int, float],
    require_oof: bool = True,
) -> tuple[list[str], list[float]]:
    print("\n" + "=" * 80)
    print("PHASE 4: OUT-OF-FOLD EVALUATION & CHECKPOINT VERIFICATION")
    print("=" * 80)

    if not requested_folds or len(requested_folds) != len(set(requested_folds)):
        raise ValueError("requested_folds must be a non-empty list of unique fold IDs")
    if any(fold not in range(5) for fold in requested_folds):
        raise ValueError(f"requested fold IDs must be in 0..4: {requested_folds}")
    if require_oof and set(best_scores) != set(requested_folds):
        raise RuntimeError(
            f"Training completed folds {sorted(best_scores)}, but requested "
            f"{sorted(requested_folds)}; refusing a partial ensemble"
        )
    fold_table = pd.read_csv(folds_csv, dtype={"StudyInstanceUID": str})
    if not {"StudyInstanceUID", "fold"}.issubset(fold_table.columns):
        raise ValueError("folds manifest must contain StudyInstanceUID and fold columns")
    if fold_table["StudyInstanceUID"].duplicated().any():
        raise ValueError("folds manifest contains duplicate StudyInstanceUID values")
    fold_values = pd.to_numeric(fold_table["fold"], errors="raise")
    if (
        fold_values.isna().any()
        or not np.equal(fold_values, fold_values.astype(int)).all()
        or not set(fold_values.astype(int)).issubset(set(range(5)))
    ):
        raise ValueError("folds manifest must assign integer fold IDs in 0..4")
    fold_table["fold"] = fold_values.astype(int)
    valid_ckpts = []
    oof_frames = []
    print("\n--- Best-checkpoint and OOF prediction verification ---")

    for fold in requested_folds:
        fold_dir = os.path.join(model_dir, f"models_fold{fold}")
        best_ckpt = os.path.join(fold_dir, f"fold{fold}_last.pt")
        oof_path = os.path.join(fold_dir, f"fold{fold}_oof.csv")
        if os.path.exists(best_ckpt):
            sz_mb = os.path.getsize(best_ckpt) / 1e6
            score_str = f"{best_scores.get(fold, float('nan')):.4f}"
            print(f"  Fold {fold}: best checkpoint [{sz_mb:.1f} MB] -> validation macro-AUC: {score_str}")
            valid_ckpts.append(best_ckpt)
            if require_oof:
                if not os.path.exists(oof_path):
                    raise FileNotFoundError(
                        f"fold {fold} completed training but its out-of-fold predictions "
                        f"are missing: {oof_path}"
                    )
                oof_fold = pd.read_csv(oof_path, dtype={"StudyInstanceUID": str})
                if oof_fold["StudyInstanceUID"].duplicated().any():
                    raise ValueError(f"fold {fold} OOF file contains duplicate study IDs")
                expected_ids = set(
                    fold_table.loc[fold_table["fold"] == fold, "StudyInstanceUID"].astype(str)
                )
                actual_ids = set(oof_fold["StudyInstanceUID"].astype(str))
                if actual_ids != expected_ids:
                    raise ValueError(
                        f"fold {fold} OOF study coverage mismatch: "
                        f"missing={len(expected_ids - actual_ids)}, "
                        f"unexpected={len(actual_ids - expected_ids)}"
                    )
                missing_predictions = {
                    f"pred_{target}" for target in config.TARGETS
                } - set(oof_fold.columns)
                if missing_predictions:
                    raise ValueError(
                        f"fold {fold} OOF file is missing prediction columns: "
                        f"{sorted(missing_predictions)}"
                    )
                predictions = oof_fold[[f"pred_{target}" for target in config.TARGETS]].to_numpy(float)
                if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
                    raise ValueError(f"fold {fold} OOF predictions contain invalid probabilities")
                oof_fold["fold"] = fold
                oof_frames.append(oof_fold)
        else:
            raise FileNotFoundError(f"Requested fold {fold} has no best checkpoint: {best_ckpt}")

    if require_oof:
        if not oof_frames:
            raise RuntimeError("Training completed but no OOF prediction files were produced")
        oof = pd.concat(oof_frames, ignore_index=True)
        if oof["StudyInstanceUID"].duplicated().any():
            raise ValueError("OOF predictions overlap across folds")
        labels = pd.read_csv(labels_csv, dtype={"StudyInstanceUID": str})
        merged = oof.merge(labels, on="StudyInstanceUID", how="left", validate="one_to_one", suffixes=("", "_label"))
        if merged[[f"{target}_weight" for target in config.TARGETS]].isna().any().any():
            raise ValueError("OOF rows contain study IDs absent from the training label table")
        from sklearn.metrics import roc_auc_score

        per_target = {}
        for target in config.TARGETS:
            gold_mask = merged[f"{target}_weight"].to_numpy(float) >= 0.99
            gold_targets = merged.loc[gold_mask, target].to_numpy(float)
            gold_predictions = merged.loc[gold_mask, f"pred_{target}"].to_numpy(float)
            if len(gold_targets) > 1 and 0 < gold_targets.sum() < len(gold_targets):
                per_target[target] = {
                    "auc": float(roc_auc_score(gold_targets, gold_predictions)),
                    "n_gold": int(len(gold_targets)),
                }
        if not per_target:
            raise RuntimeError("OOF predictions contain no gold target with both classes")
        macro_auc = float(np.mean([m["auc"] for m in per_target.values()]))
        oof_output = os.path.join(model_dir, "oof_predictions.csv")
        oof.to_csv(oof_output, index=False)
        metrics = {
            "metric": "gold-only fold-held-out macro ROC-AUC",
            "macro_auc": macro_auc,
            "selection_bias_warning": (
                "Each fold's best epoch was selected using that fold's gold validation labels; "
                "this aggregate is useful for internal comparison but is not an unbiased "
                "external estimate."
            ),
            "n_oof_studies": int(len(oof)),
            "n_gold_studies_with_oof": int(
                merged[[f"{target}_weight" for target in config.TARGETS]]
                .ge(0.99).any(axis=1).sum()
            ),
            "folds": sorted(best_scores),
            "per_target": per_target,
        }
        metrics_path = os.path.join(model_dir, "oof_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\n[OOF] Gold-only macro-AUC: {macro_auc:.4f}; metrics saved to {metrics_path}")
        print(f"[OOF] Study predictions saved to {oof_output}")

    # No calibration fit is implemented. Identity temperatures preserve ranking.
    temperatures = [1.0] * len(valid_ckpts)
    print("[CALIBRATION] No fitted calibrator available; using identity temperature T=1.")
    return valid_ckpts, temperatures


# ==============================================================================
# PHASE 5: TEST INFERENCE & SUBMISSION GENERATION (TTA; IDENTITY TEMPERATURE)
# ==============================================================================
def run_inference_phase(
    data_root: str,
    work_dir: str,
    model_ckpts: list[str],
    temperatures: list[float] | None = None,
    use_tta: bool = True,
    n_tta: int = 4,
    ensemble_checkpoints: list[str] | None = None,
    batch_size: int = 4,
    use_d4_target_weights: bool = False,
):
    print("\n" + "=" * 80)
    print("PHASE 5: TEST INFERENCE & SUBMISSION GENERATION (TTA; IDENTITY TEMPERATURE)")
    print("=" * 80)

    if ensemble_checkpoints is None:
        ensemble_checkpoints = []
    missing_checkpoints = [path for path in ensemble_checkpoints if not os.path.isfile(path)]
    if missing_checkpoints:
        raise FileNotFoundError(f"Ensemble checkpoints not found: {missing_checkpoints}")
    all_checkpoints = list(dict.fromkeys([*model_ckpts, *ensemble_checkpoints]))
    if not all_checkpoints:
        print("[WARNING] No trained model checkpoints available. Skipping inference.")
        return
    checkpoint_temperatures = dict(zip(model_ckpts, temperatures or []))
    all_temperatures = [
        checkpoint_temperatures.get(checkpoint, 1.0)
        for checkpoint in all_checkpoints
    ]

    test_csv = os.path.join(data_root, "test.csv")
    test_dir = os.path.join(data_root, "test_series") if os.path.exists(os.path.join(data_root, "test_series")) else os.path.join(data_root, "test")

    has_test_files = os.path.exists(test_csv) and os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0

    if not has_test_files:
        print(f"[INFO] Test series images not found in {data_root}.")
        print(f"[INFO] {len(all_checkpoints)} model checkpoints are verified and ready for deployment.")
        print(f"[INFO] To generate submissions on Kaggle, run:")
        print(f"       python -m src.inference.inference --root /kaggle/input/rsna-knee-abnormality-detection")
        return

    out_csv = os.path.join(work_dir, "submission.csv")
    print(
        f"Running inference with {len(all_checkpoints)} models "
        f"(TTA={use_tta}, n_tta={n_tta}, Calibrated={temperatures is not None}, "
        f"families={[os.path.basename(path) for path in all_checkpoints]})..."
    )

    sub, stats = run_inference(
        root=data_root,
        models=all_checkpoints,
        test_csv=test_csv,
        out_csv=out_csv,
        cache_dir=os.path.join(work_dir, "test_cache"),
        temperatures=all_temperatures,
        use_tta=use_tta,
        n_tta=n_tta,
        batch=batch_size,
        use_d4_target_weights=use_d4_target_weights,
    )

    # Acceptance verification
    test_df = pd.read_csv(test_csv)
    assert len(sub) == len(test_df), f"Row mismatch: submission has {len(sub)}, expected {len(test_df)}"
    assert (sub["StudyInstanceUID"].values == test_df["StudyInstanceUID"].values).all(), "StudyInstanceUID order mismatch!"
    assert not sub.isna().any().any(), "Submission contains NaN values!"
    print(f"[SUCCESS] Submission generated and verified: {out_csv} ({len(sub)} studies)")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="RSNA Knee Abnormality Detection: Unified Master Pipeline")
    parser.add_argument("--data_root", type=str, default=None, help="Path to competition dataset root")
    parser.add_argument("--work_dir", type=str, default=None, help="Working output directory")
    parser.add_argument("--model_dir", type=str, default=None, help="Directory for model checkpoints (defaults to work_dir)")
    
    # NLP Options
    parser.add_argument("--nlp_engine", type=str, default="vllm", choices=["vllm"], help="NLP engine; auto uses rules when the configured vLLM model is outside the memory policy")
    parser.add_argument("--nlp_model", type=str, default="Qwen/Qwen2.5-72B-Instruct", help="vLLM model ID for report extraction")
    parser.add_argument("--force_nlp", action="store_true", help="Force re-extraction of pseudo-labels from scratch")
    parser.add_argument(
        "--fresh_preprocessing",
        action="store_true",
        help="Rebuild DICOM index chunks and cache from scratch; existing generated cache files are overwritten",
    )
    parser.add_argument("--skip_nlp", action="store_true", help="Skip NLP extraction entirely and train with Gold labels only")

    # Training Options
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size per step")
    parser.add_argument("--grad_accum", type=int, default=config.GRAD_ACCUM, help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=config.NUM_WORKERS, help="Persistent training DataLoader workers")
    parser.add_argument("--prefetch_factor", type=int, default=config.PREFETCH_FACTOR, help="Batches prefetched per training worker")
    parser.add_argument("--eval_batch_size", type=int, default=config.EVAL_BATCH_SIZE, help="Gold validation and OOF batch size")
    parser.add_argument("--preprocess_workers", type=int, default=config.PREPROCESS_WORKERS, help="CPU workers for DICOM indexing/cache construction")
    parser.add_argument("--inference_batch_size", type=int, default=4, help="Study batch size for test inference")
    parser.add_argument("--n_windows_train", type=int, default=config.N_WINDOWS_TRAIN, help="Windows sampled per anatomical slot during training")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4", help="Comma-separated list of folds to train (e.g. '0,1,2,3,4')")
    parser.add_argument("--model_type", choices=["dinov2", "coatnet_mil", "timm_mil"], default="dinov2", help="Training architecture family")
    parser.add_argument("--variant", type=str, default=None, help="Backbone variant or timm architecture name (defaults by model family)")
    parser.add_argument(
        "--timm_pooling",
        choices=["hierarchical", "flat"],
        default="hierarchical",
        help="timm MIL pooling: target-specific windows within each slot, then target-specific slot fusion; flat preserves the legacy design",
    )
    parser.add_argument(
        "--no_slot_prior",
        action="store_true",
        help="Disable the hand-coded anatomical attention prior in the DINOv2 slot head",
    )
    parser.add_argument(
        "--random_init",
        action="store_true",
        help="Do not load pretrained weights (timm-based MIL models only)",
    )
    parser.add_argument("--skip_train", action="store_true", help="Skip model training")
    parser.add_argument(
        "--prepare_only",
        action="store_true",
        help="Run the complete preprocessing/QC gates and exit before training",
    )
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation")
    parser.add_argument(
        "--d4_target_weights",
        action="store_true",
        help="Use D4's target-specific DINO/CoAtNet weights instead of equal family-rank blending",
    )
    parser.add_argument(
        "--ensemble_checkpoints",
        nargs="+",
        default=None,
        help="Additional trained checkpoints; families receive equal rank-blend weight unless --d4_target_weights is set",
    )
    parser.add_argument("--max_ram_gb", type=float, default=config.CIRCUIT_BREAKER_MAX_RAM_GB, help="Unified-memory hard safety ceiling in GiB (default: 100 GiB)")
    args = parser.parse_args()

    if (
        args.batch_size < 1 or args.grad_accum < 1 or args.num_workers < 0
        or args.prefetch_factor < 1 or args.eval_batch_size < 1
        or args.preprocess_workers < 1 or args.inference_batch_size < 1
        or args.n_windows_train < 1 or args.epochs < 1 or args.max_ram_gb <= 0
    ):
        parser.error("batch/epoch/window/worker settings and the RAM ceiling must be positive (num_workers may be zero)")
    fold_tokens = [token.strip() for token in args.folds.split(",")]
    if any(not token.isdigit() for token in fold_tokens):
        parser.error("--folds must contain only comma-separated integer IDs")
    folds_to_run = [int(token) for token in fold_tokens]
    if (
        not folds_to_run
        or len(folds_to_run) != len(set(folds_to_run))
        or any(fold not in range(5) for fold in folds_to_run)
    ):
        parser.error("--folds must be a non-empty, comma-separated set of unique IDs from 0 through 4")

    if args.variant is None:
        args.variant = (
            "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
            if args.model_type in ("coatnet_mil", "timm_mil")
            else "dinov2-base"
        )
    if args.random_init and args.model_type == "dinov2":
        parser.error("--random_init is supported only for timm-based MIL models")
    if args.ensemble_checkpoints:
        missing_checkpoints = [p for p in args.ensemble_checkpoints if not os.path.isfile(p)]
        if missing_checkpoints:
            parser.error(f"ensemble checkpoint files not found: {missing_checkpoints}")

    if args.max_ram_gb:
        config.CIRCUIT_BREAKER_MAX_RAM_GB = args.max_ram_gb
        os.environ["RSNA_MAX_RAM_GB"] = str(args.max_ram_gb)

    global_start_time = time.time()
    work_dir = args.work_dir or os.environ.get("RSNA_OUT_DIR", os.path.join(PROJECT_ROOT, "pipeline_out"))
    os.makedirs(work_dir, exist_ok=True)
    model_dir = args.model_dir or work_dir
    if model_dir != work_dir:
        os.makedirs(model_dir, exist_ok=True)

    # Logging setup
    import logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(work_dir, f"master_pipeline_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )

    def logged_print(*p_args, **p_kwargs):
        msg = " ".join(str(a) for a in p_args)
        logging.info(msg)

    import builtins
    builtins.print = logged_print

    print("=" * 80)
    print("RSNA 2026: END-TO-END MASTER TRAINING & BUILD PIPELINE")
    print("=" * 80)
    print(f"Start Time        : {datetime.datetime.now().isoformat()}")
    print(f"Output Directory  : {work_dir}")
    print(f"Log File          : {log_file}")

    # Phase 0: Hardware & Environment
    verify_hardware_and_environment()
    data_root = resolve_data_root(args.data_root)

    # Phase 1: NLP Pseudo-Label Auto-Detection & Completion
    pseudo_csv = run_nlp_phase(
        data_root=data_root,
        work_dir=work_dir,
        skip_nlp=args.skip_nlp,
        engine=args.nlp_engine,
        model_id=args.nlp_model,
        force=(args.force_nlp or args.fresh_preprocessing),
    )

    # Phase 2: Dataset Merge & Cache Build
    labels_csv, cache_prefix, folds_csv = run_preparation(
        data_root, work_dir, pseudo_csv, force=args.fresh_preprocessing,
        workers=args.preprocess_workers,
    )

    # Check if cache is built (if download is in progress, cache_prefix is None)
    if cache_prefix is None or not os.path.exists(f"{cache_prefix}.meta.json"):
        total_elapsed = time.time() - global_start_time
        m, s = divmod(int(total_elapsed), 60)
        print("\n" + "=" * 80)
        print(f"PRE-DOWNLOAD SETUP & STAGING VERIFIED SUCCESSFULLY IN {m:02d}m {s:02d}s")
        print(f"  * Labels File : {labels_csv}")
        print(f"  * Splits File : {folds_csv}")
        print(f"  * Data Root   : {data_root}")
        print("\nReady to run full training as soon as the DICOM download completes!")
        print("To start training once download finishes:")
        print("    python src/main.py")
        print("=" * 80)
        return

    if args.prepare_only:
        print("[SUCCESS] --prepare_only: data preparation passed; training was not started.")
        return

    # Phase 3: 5-Fold Training
    best_scores = {}
    try:
        if not args.skip_train:
            best_scores = run_all_folds(
                labels_csv=labels_csv,
                cache_prefix=cache_prefix,
                folds_csv=folds_csv,
                work_dir=model_dir,
                folds_to_run=folds_to_run,
                epochs=args.epochs,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                num_workers=args.num_workers,
                prefetch_factor=args.prefetch_factor,
                eval_batch_size=args.eval_batch_size,
                n_windows_train=args.n_windows_train,
                variant=args.variant,
                model_type=args.model_type,
                timm_pooling=args.timm_pooling,
                use_slot_prior=not args.no_slot_prior,
                pretrained=not args.random_init,
            )

        # Phase 4: OOF & Checkpoints + Temperature Calibration
        valid_ckpts, fold_temperatures = run_oof_and_checkpoint_verification(
            model_dir=model_dir,
            labels_csv=labels_csv,
            folds_csv=folds_csv,
            requested_folds=folds_to_run,
            best_scores=best_scores,
            require_oof=not args.skip_train,
        )

        # Phase 5: Test Inference & Submission Generation (TTA + Calibrated)
        run_inference_phase(
            data_root=data_root,
            work_dir=model_dir,
            model_ckpts=valid_ckpts,
            temperatures=fold_temperatures,
            use_tta=(not args.no_tta),
            n_tta=4,
            ensemble_checkpoints=args.ensemble_checkpoints,
            batch_size=args.inference_batch_size,
            use_d4_target_weights=args.d4_target_weights,
        )
    except MemoryCircuitBreakerTriggered as mem_err:
        print("\n" + "=" * 80, flush=True)
        print(" [MASTER PIPELINE HALTED] UNIFIED MEMORY SAFETY CIRCUIT BREAKER ACTIVATED", flush=True)
        print(f" Current Memory Usage: {mem_err.used_gb:.2f} GB (Threshold: {mem_err.threshold_gb:.2f} GB)", flush=True)
        print(f" Pipeline stopped with {mem_err.details.get('available_gb', 0.0):.1f} GiB available; recovery headroom protected.", flush=True)
        print(" All memory allocations and GPU caches have been completely flushed.", flush=True)
        print("=" * 80, flush=True)
        print(" RESTART INSTRUCTIONS:", flush=True)
        print("   1. Drop Linux filesystem page caches on the DGX host:", flush=True)
        print("        sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches", flush=True)
        print("   2. Re-start the training pipeline:", flush=True)
        print(f"        python src/main.py --folds {args.folds} --model_dir <fresh-model-dir>", flush=True)
        print("=" * 80 + "\n", flush=True)
        sys.exit(101)

    # Phase 6: Final Summary
    total_elapsed = time.time() - global_start_time
    m, s = divmod(int(total_elapsed), 60)
    h, m = divmod(m, 60)

    print("\n" + "=" * 80)
    print(f"MASTER PIPELINE EXECUTION COMPLETED IN {h:02d}h {m:02d}m {s:02d}s")
    print(f"Artifacts Directory: {work_dir}")
    print(f"Log Output         : {log_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
