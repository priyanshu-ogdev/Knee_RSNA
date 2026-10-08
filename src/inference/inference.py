"""Inference, test-time augmentation, and family-level rank blending."""
from __future__ import annotations

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import cv2
cv2.setNumThreads(0)
import tempfile
import numpy as np
import pandas as pd
import torch

import queue
import threading

def prefetch_generator(cache, rows, cfg, n_use, batch_size):
    q = queue.Queue(maxsize=3)
    def worker():
        for lo in range(0, len(rows), batch_size):
            batch_rows = rows[lo:lo + batch_size]
            with ThreadPoolExecutor(max_workers=len(batch_rows)) as ex:
                smp = list(ex.map(lambda i: loader.make_sample(cache, i, cfg, train=False, n_use=n_use), batch_rows))
            q.put((batch_rows, smp))
        q.put(None)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    while True:
        res = q.get()
        if res is None: break
        yield res
from concurrent.futures import ThreadPoolExecutor

import src.core.config as config

from src.modeling.model import load_checkpoint
from src.data.preprocess import index as pix
from src.data.preprocess import slots as pslots
from src.data.preprocess import pipeline, cache as pcache, loader


def rank_percentile_blend(model_preds_list: list[np.ndarray]) -> np.ndarray:
    """Combines N model probability matrices [N_samples, N_targets] using Rank-Percentile Normalization.

    Reduces dependence on probability scale and gives each input prediction array
    equal rank-space weight; it does not guarantee an AUC improvement.
    """
    if not model_preds_list:
        raise ValueError("At least one prediction array is required for blending")
    if len(model_preds_list) == 1:
        return model_preds_list[0]
    n_samples, n_targets = model_preds_list[0].shape
    if any(pred.shape != (n_samples, n_targets) for pred in model_preds_list):
        raise ValueError("All model prediction arrays must share the same [samples, targets] shape")
    blended = np.zeros((n_samples, n_targets), dtype=np.float32)
    for c in range(n_targets):
        target_ranks = []
        for m_preds in model_preds_list:
            col = m_preds[:, c]
            valid = np.isfinite(col)
            ranks = np.zeros_like(col, dtype=np.float32)
            if valid.sum() > 1:
                ranks[valid] = (
                    pd.Series(col[valid])
                    .rank(method="average", pct=True)
                    .to_numpy(dtype=np.float32)
                )
            elif valid.sum() == 1:
                ranks[valid] = 0.5
            target_ranks.append(ranks)
        blended[:, c] = np.mean(target_ranks, axis=0)
    return blended


def family_rank_percentile_blend(
    per_model_preds: list[np.ndarray],
    model_families: list[str],
) -> np.ndarray:
    """Average checkpoints within architectures, then give every architecture one rank vote."""
    if len(per_model_preds) != len(model_families) or not per_model_preds:
        raise ValueError("Each prediction array must have exactly one model family")
    if any(pred.shape != per_model_preds[0].shape for pred in per_model_preds):
        raise ValueError("All model prediction arrays must have the same shape")
    families: dict[str, list[np.ndarray]] = {}
    for predictions, family in zip(per_model_preds, model_families):
        families.setdefault(family, []).append(predictions)
    family_means = [
        np.mean(np.stack(predictions), axis=0)
        for predictions in families.values()
    ]
    if len(family_means) == 1:
        return family_means[0]
    return np.mean(
        [_rank_columns_average(predictions) for predictions in family_means],
        axis=0,
        dtype=np.float32,
    )


def _rank_columns_average(values: np.ndarray) -> np.ndarray:
    """Return column-wise percentile ranks using average ties, as in the d4 notebook."""
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Rank inputs must be a finite 2D array")
    return (
        pd.DataFrame(values)
        .rank(method="average", pct=True)
        .to_numpy(dtype=np.float32)
    )


def d4_family_rank_blend(
    per_model_preds: list[np.ndarray],
    model_types: list[str],
) -> np.ndarray:
    """Apply d4-derived target weights to this pipeline's DINOv2/CoAtNet families.

    This is not the complete d4 blend: its second source is a Raptor/CoAtNet
    hybrid, whose external Raptor and auxiliary model arms are not loaded here.
    """
    if len(per_model_preds) != len(model_types) or not per_model_preds:
        raise ValueError("Each prediction array must have exactly one model family")
    if any(pred.shape != per_model_preds[0].shape for pred in per_model_preds):
        raise ValueError("All model-family prediction arrays must have the same shape")
    if per_model_preds[0].ndim != 2 or per_model_preds[0].shape[1] != len(config.TARGETS):
        raise ValueError(
            f"D4 blending requires {len(config.TARGETS)} target columns"
        )

    families: dict[str, list[np.ndarray]] = {}
    for predictions, model_type in zip(per_model_preds, model_types):
        if model_type not in {"dinov2", "coatnet_mil"}:
            raise ValueError(f"Unsupported model family for d4 blend: {model_type!r}")
        families.setdefault(model_type, []).append(predictions)

    if set(families) != {"dinov2", "coatnet_mil"}:
        raise ValueError("d4 family blend requires at least one DINOv2 and one CoAtNet checkpoint")

    family_ranks = {
        name: _rank_columns_average(np.mean(np.stack(predictions), axis=0))
        for name, predictions in families.items()
    }
    coatnet_weight = np.full(len(config.TARGETS), 0.60, dtype=np.float32)
    coatnet_overrides = {
        "ACL": 0.75,
        "Medial Meniscus": 0.80,
        "Lateral Meniscus": 1.00,
        "Lateral OA": 0.75,
        "Fracture": 0.75,
    }
    for target, weight in coatnet_overrides.items():
        coatnet_weight[config.TARGETS.index(target)] = weight

    combined = (
        (1.0 - coatnet_weight[None, :]) * family_ranks["dinov2"]
        + coatnet_weight[None, :] * family_ranks["coatnet_mil"]
    )
    return _rank_columns_average(combined)


def resolve_preprocessing_config(
    models: list[torch.nn.Module],
    cfg: config.PreCfg | None,
) -> config.PreCfg:
    """Use checkpoint preprocessing metadata and reject incompatible model ensembles."""
    checkpoint_cfgs = []
    for model in models:
        saved_cfg = getattr(model, "_rsna_preprocessing_config", None)
        if saved_cfg is not None:
            checkpoint_cfgs.append(
                config.cfg_from_dict(saved_cfg) if isinstance(saved_cfg, dict) else saved_cfg
            )

    if not checkpoint_cfgs:
        return cfg or config.get_cfg("v2")
    if len(checkpoint_cfgs) != len(models):
        raise ValueError(
            "Some model checkpoints do not contain preprocessing metadata; "
            "pass cfg explicitly to guarantee train/inference parity."
        )
    if any(saved_cfg != checkpoint_cfgs[0] for saved_cfg in checkpoint_cfgs[1:]):
        raise ValueError("Ensemble checkpoints were trained with different preprocessing configs.")
    if cfg is not None and cfg != checkpoint_cfgs[0]:
        raise ValueError("Explicit inference cfg does not match the checkpoint preprocessing config.")
    return cfg or checkpoint_cfgs[0]


def prepare_test_tables(root: str, cfg: config.PreCfg, workers: int | None = None):
    """Directory-truth index of test_series -> annotated series table, slot table, laterality, records."""
    # Robust candidate test directory detection
    test_dirs = ["test_series", "test", "test_images"]
    found_test_dir = None
    for td in test_dirs:
        if os.path.exists(os.path.join(root, td)):
            found_test_dir = td
            break

    idx = pix.build_index(root, ("test",), workers=workers, chunk=500, progress=False)
    idx = pix.attach_csv_flags(idx, root)
    idx["n_slices"] = idx["n_slices"].fillna(0) if "n_slices" in idx else 0
    ann, tab = pslots.assign_all(idx, cfg.slot_prefer_2d, cfg.slot_fs_priority, cfg.slot_csv_fallback)
    return ann, tab, pipeline.study_sides(idx), pipeline.index_to_records(ann)


def _validate_prediction_options(
    models: list[torch.nn.Module],
    temperatures: list | None,
    batch: int,
    n_tta: int | None = None,
) -> list:
    if not models:
        raise ValueError("At least one model is required for prediction")
    if batch < 1:
        raise ValueError("batch must be positive")
    if n_tta is not None and n_tta < 1:
        raise ValueError("n_tta must be positive")
    values = [1.0] * len(models) if temperatures is None else list(temperatures)
    if len(values) != len(models):
        raise ValueError(
            f"received {len(values)} temperatures for {len(models)} models"
        )
    validated = []
    for value in values:
        temperature = np.asarray(value, dtype=np.float64)
        if temperature.ndim == 0:
            if not np.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperatures must be finite and positive")
            validated.append(float(temperature))
        elif temperature.shape == (len(config.TARGETS),):
            if not np.isfinite(temperature).all() or (temperature <= 0).any():
                raise ValueError("per-target temperatures must be finite and positive")
            validated.append(temperature)
        else:
            raise ValueError(
                f"temperature must be scalar or have shape ({len(config.TARGETS)},)"
            )
    return validated


def _scale_logits(logits: torch.Tensor, temperature) -> torch.Tensor:
    if np.isscalar(temperature):
        return logits / float(temperature)
    return logits / torch.as_tensor(
        temperature, device=logits.device, dtype=logits.dtype
    )


@torch.inference_mode()
def predict_chunk_tta(
    models: list,
    cache: pcache.StudyCache,
    a: int,
    b: int,
    cfg: config.PreCfg,
    device: torch.device,
    n_use: int | None = None,
    batch: int = 4,
    temperatures: list | None = None,
    n_tta: int = 4,
    tta_seed: int = 42,
    return_models: bool = False,
) -> np.ndarray | list[np.ndarray]:
    """Average predictions over deterministic test-time views without horizontal flips.

    Batch-Inverted Cache Optimization:
    Loops over study batches on the outer axis. Each study batch is loaded from the
    memmap cache ONCE. Then N TTA views are generated in memory and inferred,
    slashing disk I/O reads by N-fold and keeping CPU memory usage strictly bounded.

    NO FLIPS: knees are laterality-canonicalised to 'left' at preprocessing time;
    horizontal flips destroy medial/lateral consistency in the slot head.

    """
    T = _validate_prediction_options(models, temperatures, batch, n_tta)
    out = []

    for lo in range(a, b, batch):
        rows = list(range(lo, min(lo + batch, b)))
        batch_tta_preds = []

        for tta_pass in range(n_tta):
            use_aug = (tta_pass > 0)
            rng_seed = tta_seed + tta_pass * 1000

            def _make(i):
                # Give each study/view an independent, reproducible augmentation.
                row_rng = np.random.default_rng(rng_seed + i)
                return loader.make_sample(
                    cache, i, cfg,
                    train=False,          # always use eval-mode window selection (evenly spaced)
                    n_use=n_use,
                    rng=row_rng,
                    aug=use_aug,          # augmentation applied only for TTA passes 1+
                )

            with ThreadPoolExecutor(max_workers=len(rows)) as ex:
                smp = list(ex.map(_make, rows))

            imgs = torch.from_numpy(np.stack([s[0] for s in smp]))
            slot = torch.from_numpy(np.stack([s[1] for s in smp])).float()
            wm   = torch.from_numpy(np.stack([s[2] for s in smp])).float()
            del smp
            
            if device.type == "cuda":
                # T4 GPU (Turing) Efficiency Fixes
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                # torch.backends.cuda.enable_math_sdp(False) # Removed to prevent hard crashes
                imgs = imgs.pin_memory()
                slot = slot.pin_memory()
                wm = wm.pin_memory()
            imgs = imgs.to(device, non_blocking=True)
            slot = slot.to(device, non_blocking=True)
            wm   = wm.to(device, non_blocking=True)

            ps = []
            for m, t_val in zip(models, T):
                use_bf16 = torch.cuda.is_bf16_supported() if device.type == "cuda" else False
                ctx = torch.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16,
                                     enabled=(device.type == "cuda"))
                with ctx:
                    logits = m(imgs, slot, wm).float()
                    scaled = _scale_logits(logits, t_val)
                    ps.append(torch.sigmoid(scaled).cpu().numpy())

            # Model ensemble mean for this TTA view: [batch_size, n_targets]
            batch_tta_preds.append(ps)

        # TTA ensemble mean across passes for this batch
        m_batch = [np.mean([batch_tta_preds[p][m_idx] for p in range(n_tta)], axis=0) for m_idx in range(len(models))]
        out.append(m_batch)

    if not out:
        empty = np.zeros((0, len(config.TARGETS)), dtype=np.float32)
        return [empty for _ in models] if return_models else empty
    per_model_preds = [np.concatenate([out[b_idx][m_idx] for b_idx in range(len(out))], axis=0) for m_idx in range(len(models))]
    if return_models:
        return per_model_preds
    return np.mean(per_model_preds, axis=0)


@torch.inference_mode()
def predict_chunk(
    models: list[torch.nn.Module],
    cache: pcache.StudyCache,
    a: int,
    b: int,
    cfg: config.PreCfg,
    device: torch.device,
    n_use: int | None = None,
    batch: int = 4,
    temperatures: list | None = None,
    return_models: bool = False,
) -> np.ndarray | list[np.ndarray]:
    """Predict a chunk of studies using an ensemble of models (e.g., 5 folds).

    If temperatures are provided, logits are scaled by T before sigmoid.
    Returns: [b-a, n_targets] average probabilities.
    """
    T = _validate_prediction_options(models, temperatures, batch)
    out = []

    for lo in range(a, b, batch):
        rows = list(range(lo, min(lo + batch, b)))
        with ThreadPoolExecutor(max_workers=len(rows)) as ex:
            smp = list(ex.map(lambda i: loader.make_sample(cache, i, cfg, train=False, n_use=n_use), rows))

        imgs = torch.from_numpy(np.stack([s[0] for s in smp]))
        slot = torch.from_numpy(np.stack([s[1] for s in smp])).float()
        wm   = torch.from_numpy(np.stack([s[2] for s in smp])).float()
        del smp

        if device.type == "cuda":
            imgs, slot, wm = imgs.pin_memory(), slot.pin_memory(), wm.pin_memory()
        imgs = imgs.to(device, non_blocking=device.type == "cuda")
        slot = slot.to(device, non_blocking=device.type == "cuda")
        wm = wm.to(device, non_blocking=device.type == "cuda")

        ps = []
        for m, t_val in zip(models, T):
            use_bf16 = torch.cuda.is_bf16_supported() if device.type == "cuda" else False
            ctx = torch.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16,
                                 enabled=(device.type == "cuda"))
            with ctx:
                logits = m(imgs, slot, wm).float()
                scaled = _scale_logits(logits, t_val)
                ps.append(torch.sigmoid(scaled).cpu().numpy())

        out.append(ps)
    if not out:
        empty = np.zeros((0, len(config.TARGETS)), dtype=np.float32)
        return [empty for _ in models] if return_models else empty
    per_model_preds = [np.concatenate([out[b_idx][m_idx] for b_idx in range(len(out))], axis=0) for m_idx in range(len(models))]
    if return_models:
        return per_model_preds
    return np.mean(per_model_preds, axis=0)


def run_inference(
    root: str,
    models: list[torch.nn.Module | str],
    test_csv: str | None = None,
    cfg: config.PreCfg | None = None,
    out_csv: str = "submission.csv",
    cache_dir: str | None = None,
    workers: int | None = None,
    chunk: int = 64,
    n_use: int | None = None,
    device: torch.device | None = None,
    temperatures: list | None = None,
    use_tta: bool = True,
    n_tta: int = 4,
    batch: int = 4,
    use_d4_target_weights: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Build the cache and run predictions in a pipelined fashion."""
    _validate_prediction_options(models, temperatures, batch, n_tta if use_tta else None)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Auto-load checkpoint files if paths were passed instead of module instances
    loaded_models = [load_checkpoint(m, device) if isinstance(m, str) else m for m in models]
    model_types = [
        getattr(model, "_rsna_model_type", getattr(model, "model_type", "dinov2"))
        for model in loaded_models
    ]
    model_families = [
        getattr(model, "_rsna_model_family", model_type)
        for model, model_type in zip(loaded_models, model_types)
    ]
    unknown_types = set(model_types) - {"dinov2", "coatnet_mil", "timm_mil"}
    if unknown_types:
        raise ValueError(f"Unsupported model families in inference ensemble: {sorted(unknown_types)}")
    if use_d4_target_weights and set(model_types) != {"dinov2", "coatnet_mil"}:
        raise ValueError(
            "D4 target weights require exactly the DINOv2 and CoAtNet model families"
        )
    cfg = resolve_preprocessing_config(loaded_models, cfg)

    ann, tab, sides, records = prepare_test_tables(root, cfg, workers)
    studies = list(tab.index)

    # Reorder to match test.csv if provided (mandatory for Kaggle submission)
    order = None
    if test_csv and os.path.exists(test_csv):
        order = pd.read_csv(test_csv)["StudyInstanceUID"].astype(str).str.strip().tolist()
        seen = set(order)
        studies = list(order) + [s for s in tab.index if s not in seen]

    cache_dir = cache_dir or ("/kaggle/temp" if os.path.isdir("/kaggle/temp") else tempfile.gettempdir())
    prefix = os.path.join(cache_dir, "test_cache")
    slot_rows = {s: tab.loc[s].to_dict() if s in tab.index else {} for s in studies}

    state = {"moved": False}
    multi_model = len(loaded_models) > 1
    if multi_model:
        per_model_preds = [np.full((len(studies), len(config.TARGETS)), np.nan, np.float32) for _ in loaded_models]
    preds = np.full((len(studies), len(config.TARGETS)), np.nan, np.float32)

    def on_ready(cache: pcache.StudyCache, a: int, b: int):
        # Workers were forked at the first submit, i.e., before any CUDA call in this process.
        if not state["moved"]:
            for i, m in enumerate(loaded_models):
                # 1. Native FP16 Casting (Faster than autocast overhead)
                m = m.eval()
                if device.type == "cuda":
                    m = m.half()
                m = m.to(device, non_blocking=device.type == "cuda")
                
                # 2. PyTorch 2.x Compiler with CUDA Graphs
                try:
                    import torch._dynamo
                    torch._dynamo.config.suppress_errors = True
                    m = torch.compile(m, mode="reduce-overhead", fullgraph=False)
                    print(f"[OPTIMIZE] Model {i} successfully compiled with reduce-overhead.")
                except Exception as e:
                    print(f"[OPTIMIZE] Compiler bypassed: {e}")
                
                loaded_models[i] = m
                # Multi-GPU check: only wrap in DataParallel if batch size is large enough to divide evenly
                if device.type == "cuda" and torch.cuda.device_count() > 1 and batch >= torch.cuda.device_count() * 2:
                    loaded_models[i] = torch.nn.DataParallel(m)
            state["moved"] = True
        if multi_model:
            if use_tta:
                chunk_m = predict_chunk_tta(loaded_models, cache, a, b, cfg, device, n_use, 
                                            batch=batch, temperatures=temperatures, n_tta=n_tta, return_models=True)
            else:
                chunk_m = predict_chunk(loaded_models, cache, a, b, cfg, device, n_use, 
                                        batch=batch, temperatures=temperatures, return_models=True)
            for m_idx, chunk_arr in enumerate(chunk_m):
                per_model_preds[m_idx][a:b] = chunk_arr
        else:
            if use_tta:
                preds[a:b] = predict_chunk_tta(loaded_models, cache, a, b, cfg, device, n_use, 
                                               batch=batch, temperatures=temperatures, n_tta=n_tta)
            else:
                preds[a:b] = predict_chunk(loaded_models, cache, a, b, cfg, device, n_use, 
                                           batch=batch, temperatures=temperatures)

    cache, stats = pcache.build_cache(
        prefix, studies, slot_rows, records, sides, cfg,
        workers=workers, resume=False, order="seq", chunk=chunk, on_ready=on_ready
    )
    print("cache stats:", stats)

    # Normalize model-family scales before the optional family ensemble.
    if multi_model:
        if use_d4_target_weights:
            preds = d4_family_rank_blend(per_model_preds, model_types)
            print("[BLEND] Applied explicitly requested D4-derived target-weighted rank blend.")
        else:
            preds = family_rank_percentile_blend(per_model_preds, model_families)

    # Impute missing studies (fallback to median so submission never fails)
    empty = np.asarray(cache.slot).sum(1) == 0
    fill = np.nanmedian(preds[~empty], axis=0) if (~empty).any() else np.full(len(config.TARGETS), 0.5)
    preds[empty] = fill
    preds = np.where(np.isfinite(preds), preds, fill)

    # A single family remains on its probability scale; multi-family blends are
    # percentile-rank scores and should not be described as calibrated probabilities.
    sub = pd.DataFrame(preds, columns=config.TARGETS)
    sub.insert(0, "StudyInstanceUID", studies)
    if order is not None:
        # Mandatory Kaggle submission rule: exact 1-to-1 match with test.csv rows and order
        sub = sub.set_index("StudyInstanceUID").reindex(order).reset_index()

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    sub.to_csv(out_csv, index=False)

    return sub, stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run trained RSNA knee checkpoints and write a Kaggle submission."
    )
    parser.add_argument("--root", required=True, help="Competition dataset root containing test.csv and test_series/")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Trained fold checkpoint paths; model families receive equal rank-blend weight by default",
    )
    parser.add_argument("--out", default="submission.csv", help="Output submission CSV")
    parser.add_argument("--cache_dir", default=None, help="Optional test cache directory")
    parser.add_argument("--batch", type=int, default=8, help="Inference batch size")
    parser.add_argument("--n_tta", type=int, default=4, help="Number of test-time views")
    parser.add_argument("--no_tta", action="store_true", help="Disable test-time augmentation")
    parser.add_argument(
        "--d4_target_weights",
        action="store_true",
        help="Opt into D4's target-specific DINO/CoAtNet weights instead of equal family-rank blending",
    )
    args = parser.parse_args()

    missing = [path for path in args.checkpoints if not os.path.isfile(path)]
    if missing:
        parser.error(f"checkpoint files not found: {missing}")
    if args.batch <= 0 or args.n_tta <= 0:
        parser.error("--batch and --n_tta must be positive")
    test_csv = os.path.join(args.root, "test.csv")
    if not os.path.isfile(test_csv):
        parser.error(f"test.csv not found under --root: {args.root}")

    submission, stats = run_inference(
        root=args.root,
        models=args.checkpoints,
        test_csv=test_csv,
        out_csv=args.out,
        cache_dir=args.cache_dir,
        use_tta=not args.no_tta,
        n_tta=args.n_tta,
        batch=args.batch,
        use_d4_target_weights=args.d4_target_weights,
    )
    test_ids = pd.read_csv(test_csv, dtype={"StudyInstanceUID": str})[
        "StudyInstanceUID"
    ].astype(str).str.strip().tolist()
    if submission["StudyInstanceUID"].astype(str).tolist() != test_ids:
        raise RuntimeError("Generated submission study IDs/order do not match test.csv")
    if submission.isna().any().any():
        raise RuntimeError("Generated submission contains missing values")
    print(f"[SUCCESS] Submission: {args.out} ({len(submission)} studies); cache={stats}")


if __name__ == "__main__":
    main()
