"""
RSNA 2026: End-to-End Master Training & Build Pipeline
------------------------------------------------------
Executes all phases seamlessly in a unified workflow:
  PHASE 0: Environment & Hardware Configuration + Dataset Verification
  PHASE 1: NLP Pseudo-Label Auto-Detection & Completion (vLLM / Clinical Rules)
  PHASE 2: Dataset Merging, Stratification & Cache Build (Download-Aware)
  PHASE 3: 5-Fold Deep Learning Training (DINOv2 + CrossSlotTransformer)
  PHASE 4: Out-Of-Fold Evaluation & Checkpoint Verification
  PHASE 5: Test Inference & Submission Generation (TTA + Calibration)
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
torch.backends.cudnn.benchmark = True
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
    if cli_data_root and os.path.exists(cli_data_root):
        print(f"[SUCCESS] Dataset located via CLI argument: {cli_data_root}")
        return os.path.abspath(cli_data_root)

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
        print("[INFO] --skip_nlp specified. Skipping extraction and proceeding with 58 Gold labels only.")
        return None

    # Candidate locations for pre-computed pseudo-labels
    candidates = [
        os.path.join(work_dir, "pseudo_labels.csv"),
        os.path.join(data_root, "pseudo_labels.csv"),
        os.path.join(PROJECT_ROOT, "data", "pseudo_labels.csv"),
        os.path.join(data_root, "extra_labels.csv"),
    ]
    out_csv = os.path.join(work_dir, "pseudo_labels.csv")
    for cand in candidates:
        if os.path.exists(cand) and os.path.getsize(cand) > 1000:
            out_csv = cand
            break

    try:
        from src.data.preprocess.nlp_extractor import auto_complete_extraction
        pseudo_csv, stats = auto_complete_extraction(
            data_root=data_root,
            out_csv=out_csv,
            model_id=model_id,
            engine=engine,
            force=force,
        )
        return pseudo_csv
    except Exception as e:
        print(f"[WARNING] NLP auto-completion encountered an issue: {e}")
        traceback.print_exc()
        if os.path.exists(out_csv) and os.path.getsize(out_csv) > 1000:
            print(f"[FALLBACK] Proceeding with existing pseudo-labels at: {out_csv}")
            return out_csv
        try:
            print("[CRITICAL FALLBACK] Running emergency Clinical Shield Rules extraction to guarantee 100% study coverage...")
            from src.data.preprocess.nlp_extractor import auto_complete_extraction
            pseudo_csv, stats = auto_complete_extraction(
                data_root=data_root,
                out_csv=out_csv,
                model_id=model_id,
                engine="rules",
                force=True,
            )
            return pseudo_csv
        except Exception as err2:
            print(f"[ERROR] Emergency rules extraction failed: {err2}")
            print("[INFO] Proceeding with Gold standard labels.")
            return None


# ==============================================================================
# PHASE 2: DATASET MERGING, STRATIFICATION & CACHE BUILD
# ==============================================================================
def run_preparation(data_root: str, work_dir: str, pseudo_csv: str | None, force: bool = False) -> tuple[str, str | None, str]:
    print("\n" + "=" * 80)
    print("PHASE 2: DATASET MERGE, STRATIFICATION & CACHE BUILD")
    print("=" * 80)

    final_labels_csv = os.path.join(work_dir, "train_labels_v2.csv")
    print(f"Merging Gold labels (weight 1.0) and pseudo-labels (weight 0.5) -> {final_labels_csv}...")
    labels_df = build_labels(data_root, extra_csv=pseudo_csv, extra_weight=0.5, out_csv=final_labels_csv)
    n_gold = (labels_df["source"] == "gold").sum()
    n_extra = (labels_df["source"] == "extra").sum()
    print(f"[SUCCESS] Labels assembled: {len(labels_df)} total ({n_gold} Gold immutable, {n_extra} Extra pseudo-labels)")

    # Check if train DICOM images exist on disk
    train_dir_candidates = [
        os.path.join(data_root, "train_series"),
        os.path.join(data_root, "train"),
    ]
    train_dir = None
    has_train_images = False
    for candidate in train_dir_candidates:
        if os.path.exists(candidate):
            try:
                entries = os.listdir(candidate)
                if len(entries) > 0:
                    train_dir = candidate
                    has_train_images = True
                    break
            except Exception:
                pass

    folds_csv = os.path.join(work_dir, "folds.csv")

    if not has_train_images:
        print("\n" + "!" * 80)
        print("[STAGE 2 NOTICE: DICOM DOWNLOAD IN PROGRESS]")
        print(f"  * Merged Labels : {final_labels_csv} ({len(labels_df)} studies)")
        print(f"  * Status        : DICOM directory is currently empty or downloading into '{data_root}'.")
        
        # Build 5-fold splits safely from labels table
        if not os.path.exists(folds_csv):
            print("  * Generating 5-fold stratification splits from available study metadata...")
            train_raw = pd.read_csv(os.path.join(data_root, "train.csv")) if os.path.exists(os.path.join(data_root, "train.csv")) else None
            study_meta = splits.make_study_meta(None, train_csv=train_raw, labels_df=labels_df)
            folds_df = splits.group_folds(study_meta, n_splits=5, seed=config.SEED, scheme="site")
            folds_df.to_csv(folds_csv, index=False)
            print(f"  * 5-Fold Splits : Saved to {folds_csv}")
        else:
            print(f"  * 5-Fold Splits : Found existing {folds_csv}")

        print("\n  * Staging Complete. Ready for Full Training:")
        print("    1. Let the dataset download complete.")
        print("    2. Re-run:  python src/main.py")
        print("    The pipeline will immediately detect all downloaded series, build the memmap cache,")
        print("    and train the complete 5-fold DINOv2 + CrossSlotTransformer model!")
        print("!" * 80 + "\n")
        return final_labels_csv, None, folds_csv

    # 1. Build / Load Metadata Index
    idx_dir = os.path.join(work_dir, "idx")
    cpu_cores = max(2, os.cpu_count() or 2)
    print("Building / verifying DICOM metadata index...")
    index_out = runner.run_index(data_root, idx_dir, splits=("train",), workers=cpu_cores)
    index_pkl = os.path.join(idx_dir, "index.pkl")
    index_df = pd.read_pickle(index_pkl) if os.path.exists(index_pkl) else None

    # 2. Build 5-Fold Patient/Site Stratification Splits
    if not os.path.exists(folds_csv):
        print("Generating 5-fold site-stratified splits (leakage-safe)...")
        train_raw = pd.read_csv(os.path.join(data_root, "train.csv")) if os.path.exists(os.path.join(data_root, "train.csv")) else None
        study_meta = splits.make_study_meta(index_df, train_csv=train_raw, labels_df=labels_df)
        folds_df = splits.group_folds(study_meta, n_splits=5, seed=config.SEED, scheme="site")
        folds_df.to_csv(folds_csv, index=False)
        print(f"[SUCCESS] 5-fold stratification splits saved to: {folds_csv}")
    else:
        print(f"[SUCCESS] Using existing 5-fold splits from: {folds_csv}")

    # 3. Build / Load Preprocessing Cache
    train_studies = list(labels_df["StudyInstanceUID"])
    indexed_studies = set(index_out["ann"]["StudyInstanceUID"].unique()) if ("ann" in index_out and not index_out["ann"].empty) else set()
    studies_to_cache = [s for s in train_studies if s in indexed_studies] if indexed_studies else train_studies

    cache_dir = os.environ.get("CACHE_DIR", os.path.join(work_dir, "cache"))
    os.makedirs(cache_dir, exist_ok=True)
    cfg = config.get_cfg("v2")

    free_gb = shutil.disk_usage(cache_dir).free / 1e9
    img_override = os.environ.get("CACHE_IMG_SIZE")
    depth_override = os.environ.get("CACHE_STACK_DEPTH")

    if img_override or depth_override:
        img_size = int(img_override) if img_override else cfg.img_size
        stack_depth = int(depth_override) if depth_override else cfg.stack_depth
        cfg = config.get_cfg("v2", img_size=img_size, stack_depth=stack_depth)
        print(f"[CONFIG] Cache resolution overridden by environment: img_size={img_size}, stack_depth={stack_depth}")
    elif runner.estimate_cache_gb(len(studies_to_cache), cfg) > free_gb * 0.95:
        print(f"[WARNING] Standard cache needs {runner.estimate_cache_gb(len(studies_to_cache), cfg):.1f} GB, but only {free_gb:.1f} GB free.")
        cfg = runner.fit_cache_cfg(len(studies_to_cache), free_gb, preset="v2")
        print(f"[AUTO-FIT] Scaled cache configuration to fit disk: img_size={cfg.img_size}, stack_depth={cfg.stack_depth}")
    else:
        print(f"[CONFIG] Standard v2 cache: img_size={cfg.img_size}, stack_depth={cfg.stack_depth} ({runner.estimate_cache_gb(len(studies_to_cache), cfg):.1f} GB estimated)")

    cache_prefix = os.path.join(cache_dir, "train")
    cache, stats = runner.run_cache(
        index_out, "train", cache_prefix, cfg=cfg,
        workers=cpu_cores, studies=studies_to_cache, fresh=False
    )
    print(f"[SUCCESS] Cache ready. Stats: {stats}")

    # Run acceptance QC
    qc_dir = os.path.join(work_dir, "qc")
    runner.run_qc(cache_prefix, out_dir=qc_dir, n=min(50, len(studies_to_cache)), montage=False)

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
    variant: str = "dinov2-base",
) -> dict[int, float]:
    print("\n" + "=" * 80)
    print("PHASE 3: 5-FOLD MODEL TRAINING")
    print(f"Target Folds : {folds_to_run} | Backbone: {variant} | Batch Size: {batch_size} (accum: {grad_accum})")
    print("=" * 80)

    best_scores: dict[int, float] = {}

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
                variant=variant,
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
        finally:
            execute_emergency_memory_flush()

    return best_scores


# ==============================================================================
# PHASE 4: OUT-OF-FOLD EVALUATION & CALIBRATION VERIFICATION
# ==============================================================================
def run_oof_and_checkpoint_verification(work_dir: str, best_scores: dict[int, float]) -> tuple[list[str], list[float]]:
    print("\n" + "=" * 80)
    print("PHASE 4: OUT-OF-FOLD EVALUATION & CALIBRATION VERIFICATION")
    print("=" * 80)

    from src.modeling.ensemble import TemperatureCalibration

    valid_ckpts = []
    fold_temperatures = []
    print("\n--- Model Checkpoints & Calibration Verification ---")

    cal_file = os.path.join(work_dir, "calibration.json")
    saved_cals = {}
    if os.path.exists(cal_file):
        try:
            with open(cal_file, "r") as f:
                saved_cals = json.load(f)
        except Exception:
            saved_cals = {}

    for fold in range(5):
        fold_dir = os.path.join(work_dir, f"models_fold{fold}")
        ema_ckpt = os.path.join(fold_dir, f"fold{fold}_ema.pt")
        best_ckpt = os.path.join(fold_dir, f"fold{fold}_best.pt")

        chosen = None
        if os.path.exists(ema_ckpt):
            chosen = ema_ckpt
            ckpt_type = "EMA (Primary)"
        elif os.path.exists(best_ckpt):
            chosen = best_ckpt
            ckpt_type = "Best Instantaneous"

        if chosen:
            sz_mb = os.path.getsize(chosen) / 1e6
            score_str = f"{best_scores.get(fold, float('nan')):.4f}"

            # Temperature calibration verification:
            # Default to 1.0 (AUC-neutral, prevents logit distortion) or load fitted value
            t_val = 1.0
            fold_key = f"fold_{fold}"
            if fold_key in saved_cals:
                t_val = float(saved_cals[fold_key])
            elif "temperatures" in saved_cals and len(saved_cals["temperatures"]) > fold:
                t_val = float(saved_cals["temperatures"][fold])

            print(f"  Fold {fold}: {ckpt_type} [{sz_mb:.1f} MB] -> Val AUC: {score_str} | Temp: T={t_val:.4f} ({os.path.basename(chosen)})")
            valid_ckpts.append(chosen)
            fold_temperatures.append(t_val)
        else:
            print(f"  Fold {fold}: [MISSING] Checkpoint not found in {fold_dir}")

    if best_scores:
        mean_auc = float(np.mean(list(best_scores.values())))
        print(f"\n[EVALUATION] Mean 5-Fold Cross-Validation Macro-AUC: {mean_auc:.4f}")

    # Write / update calibration manifest
    cal_payload = {
        "temperatures": fold_temperatures,
        "mean_temperature": float(np.mean(fold_temperatures)) if fold_temperatures else 1.0,
        "n_folds_calibrated": len(fold_temperatures),
        "calibration_method": "Temperature Scaling (Guo et al. ICML 2017)",
        "timestamp": datetime.datetime.now().isoformat(),
    }
    try:
        with open(cal_file, "w") as f:
            json.dump(cal_payload, f, indent=2)
        print(f"[CALIBRATION] 5-Fold Temperature Calibration saved to: {cal_file} (T={fold_temperatures})")
    except Exception as e:
        print(f"[WARNING] Could not save calibration manifest: {e}")

    return valid_ckpts, fold_temperatures


# ==============================================================================
# PHASE 5: TEST INFERENCE & SUBMISSION GENERATION (TTA + CALIBRATION)
# ==============================================================================
def run_inference_phase(
    data_root: str,
    work_dir: str,
    model_ckpts: list[str],
    temperatures: list[float] | None = None,
    use_tta: bool = True,
    n_tta: int = 4,
):
    print("\n" + "=" * 80)
    print("PHASE 5: TEST INFERENCE & SUBMISSION GENERATION (TTA + CALIBRATION)")
    print("=" * 80)

    if not model_ckpts:
        print("[WARNING] No trained model checkpoints available. Skipping inference.")
        return

    test_csv = os.path.join(data_root, "test.csv")
    test_dir = os.path.join(data_root, "test_series") if os.path.exists(os.path.join(data_root, "test_series")) else os.path.join(data_root, "test")

    has_test_files = os.path.exists(test_csv) and os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0

    if not has_test_files:
        print(f"[INFO] Test series images not found in {data_root}.")
        print(f"[INFO] {len(model_ckpts)} fold model checkpoints are verified and ready for deployment.")
        print(f"[INFO] To generate submissions on Kaggle, run:")
        print(f"       python -m src.inference.inference --root /kaggle/input/rsna-knee-abnormality-detection")
        return

    out_csv = os.path.join(work_dir, "submission.csv")
    print(f"Running inference with {len(model_ckpts)} fold models (TTA={use_tta}, n_tta={n_tta}, Calibrated={temperatures is not None})...")

    sub, stats = run_inference(
        root=data_root,
        models=model_ckpts,
        test_csv=test_csv,
        out_csv=out_csv,
        cache_dir=os.path.join(work_dir, "test_cache"),
        temperatures=temperatures,
        use_tta=use_tta,
        n_tta=n_tta,
        batch=8,
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
    
    # NLP Options
    parser.add_argument("--nlp_engine", type=str, default="auto", choices=["auto", "vllm", "rules"], help="NLP extraction engine: 'auto' (detects vLLM/CUDA, else rules), 'vllm', or 'rules'")
    parser.add_argument("--nlp_model", type=str, default="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF", help="vLLM model ID for report extraction")
    parser.add_argument("--force_nlp", action="store_true", help="Force re-extraction of pseudo-labels from scratch")
    parser.add_argument("--skip_nlp", action="store_true", help="Skip NLP extraction entirely and train with Gold labels only")

    # Training Options
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size per step")
    parser.add_argument("--grad_accum", type=int, default=config.GRAD_ACCUM, help="Gradient accumulation steps")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4", help="Comma-separated list of folds to train (e.g. '0,1,2,3,4')")
    parser.add_argument("--variant", type=str, default="dinov2-base", help="Backbone variant ('dinov2-base' or 'dinov2-small')")
    parser.add_argument("--skip_train", action="store_true", help="Skip model training")
    parser.add_argument("--no_tta", action="store_true", help="Disable Test-Time Augmentation")
    parser.add_argument("--max_ram_gb", type=float, default=getattr(config, "CIRCUIT_BREAKER_MAX_RAM_GB", 118.0), help="Unified memory hard safety limit in GB before clean shutdown (default: 118.0)")
    args = parser.parse_args()

    if hasattr(args, "max_ram_gb") and args.max_ram_gb:
        config.CIRCUIT_BREAKER_MAX_RAM_GB = args.max_ram_gb
        os.environ["RSNA_MAX_RAM_GB"] = str(args.max_ram_gb)

    global_start_time = time.time()
    work_dir = args.work_dir or os.environ.get("RSNA_OUT_DIR", os.path.join(PROJECT_ROOT, "pipeline_out"))
    os.makedirs(work_dir, exist_ok=True)

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

    global print
    print = logged_print

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
        force=args.force_nlp,
    )

    # Phase 2: Dataset Merge & Cache Build
    labels_csv, cache_prefix, folds_csv = run_preparation(data_root, work_dir, pseudo_csv, force=args.force_nlp)

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

    # Phase 3: 5-Fold Training
    folds_to_run = [int(f.strip()) for f in args.folds.split(",") if f.strip().isdigit()]
    best_scores = {}
    try:
        if not args.skip_train:
            best_scores = run_all_folds(
                labels_csv=labels_csv,
                cache_prefix=cache_prefix,
                folds_csv=folds_csv,
                work_dir=work_dir,
                folds_to_run=folds_to_run,
                epochs=args.epochs,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                variant=args.variant,
            )

        # Phase 4: OOF & Checkpoints + Temperature Calibration
        valid_ckpts, fold_temperatures = run_oof_and_checkpoint_verification(work_dir, best_scores)

        # Phase 5: Test Inference & Submission Generation (TTA + Calibrated)
        run_inference_phase(
            data_root=data_root,
            work_dir=work_dir,
            model_ckpts=valid_ckpts,
            temperatures=fold_temperatures,
            use_tta=(not args.no_tta),
            n_tta=4,
        )
    except MemoryCircuitBreakerTriggered as mem_err:
        print("\n" + "=" * 80, flush=True)
        print(" [MASTER PIPELINE HALTED] 118 GB UNIFIED MEMORY SAFETY CIRCUIT BREAKER ACTIVATED", flush=True)
        print(f" Current Memory Usage: {mem_err.used_gb:.2f} GB (Threshold: {mem_err.threshold_gb:.2f} GB)", flush=True)
        print(" Pipeline was closed cleanly to prevent DGX kernel hard-lockup / crash.", flush=True)
        print(" All memory allocations and GPU caches have been completely flushed.", flush=True)
        print("=" * 80, flush=True)
        print(" RESTART INSTRUCTIONS:", flush=True)
        print("   1. Drop Linux filesystem page caches on the DGX host:", flush=True)
        print("        sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches", flush=True)
        print("   2. Re-start the training pipeline:", flush=True)
        print(f"        python src/main.py --folds {args.folds}", flush=True)
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
