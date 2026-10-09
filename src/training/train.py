"""Gold-validated multi-label training with memory-bounded input pipelines."""
from __future__ import annotations
import math

import os
import time
import dataclasses
import numpy as np
import pandas as pd
import psutil
# CRITICAL CPU THRASHING FIX:
# Limit OpenCV and NumPy internal threading. PyTorch DataLoader spawns multiple processes
# OpenCV spawning threads equal to the core count inside each worker can thrash the CPU
# and bottleneck the data pipeline.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

import cv2
cv2.setNumThreads(0)

import torch
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
if hasattr(torch.backends.cuda.matmul, "allow_fp16_reduced_precision_reduction"):
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
if hasattr(torch.backends.cuda, "enable_flash_sdp"):
    torch.backends.cuda.enable_flash_sdp(True)
if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
    torch.backends.cuda.enable_mem_efficient_sdp(True)

import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel

import src.core.config as config
from src.modeling.losses import build_loss
from src.modeling.model import build_model
from src.data.dataset import RSNADataset
from src.data.preprocess.cache import cfg_of as cache_cfg


# ────────────────────────────────────── Memory Circuit Breaker ────────────────
class MemoryCircuitBreakerTriggered(RuntimeError):
    """Raised when unified-memory headroom reaches the configured hard limit."""
    def __init__(self, used_gb: float, threshold_gb: float, stage: str, details: dict | None = None):
        self.used_gb = used_gb
        self.threshold_gb = threshold_gb
        self.stage = stage
        self.details = details or {}
        super().__init__(
            f"Unified Memory Circuit Breaker Triggered: {used_gb:.2f} GB used "
            f"(threshold: {threshold_gb:.2f} GB) during '{stage}'"
        )


def check_memory_circuit_breaker(
    stage: str,
    threshold_gb: float | None = None,
) -> tuple[float, float]:
    """Stop before the host loses the recovery headroom needed by the OS."""
    limit = threshold_gb if threshold_gb is not None else config.CIRCUIT_BREAKER_MAX_RAM_GB
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    available_gb = mem.available / (1024 ** 3)
    min_available = getattr(config, "MIN_AVAILABLE_RAM_GB", 20.0)
    if used_gb >= limit or available_gb <= min_available:
        details = {
            "used_gb": used_gb,
            "total_gb": total_gb,
            "percent": mem.percent,
            "available_gb": available_gb,
        }
        raise MemoryCircuitBreakerTriggered(used_gb, limit, stage, details)
    return used_gb, total_gb


def execute_emergency_memory_flush(active_loaders: list | None = None):
    """Completely purges all GPU allocations, PyTorch caches, worker pools, and OS glibc arenas."""
    if active_loaders:
        for loader in active_loaders:
            if loader is not None:
                try:
                    _shutdown_loader(loader)
                except Exception:
                    pass
                try:
                    del loader
                except Exception:
                    pass
    import gc
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        if hasattr(torch.cuda, "ipc_collect"):
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def _shutdown_loader(loader) -> None:
    """Explicitly stop persistent DataLoader workers before releasing a fold."""
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
        loader._iterator = None


def _atomic_torch_save(payload: dict, path: str) -> None:
    temporary_path = f"{path}.tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def _loader_batch_bytes(cfg, windows: int, batch_size: int) -> int:
    return (
        config.N_SLOTS * windows * cfg.group * cfg.img_size * cfg.img_size
        * batch_size
    )


def handle_memory_circuit_breaker(
    exc: MemoryCircuitBreakerTriggered,
    fold: int,
    epoch: int | None = None,
    step: int | None = None,
    model: nn.Module | None = None,
    cfg = None,
    variant: str = "dinov2-base",
    use_cross_slot: bool = True,
    model_config: dict | None = None,
    out_dir: str = ".",
    active_loaders: list | None = None,
):
    """Saves emergency checkpoint, prints user-visible alert & restart banner, flushes memory."""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "!" * 80, flush=True)
    print(f" [CRITICAL SAFETY BREAKER] UNIFIED MEMORY BREACHED {exc.threshold_gb:.1f} GB LIMIT!", flush=True)
    print(f" Timestamp : {timestamp}", flush=True)
    print(f" Memory    : {exc.used_gb:.2f} GB used / {exc.details.get('total_gb', 128.0):.1f} GB total ({exc.details.get('percent', 0.0):.1f}%)", flush=True)
    print(f" Location  : Fold {fold} | Epoch {epoch if epoch is not None else '?'} | Step {step if step is not None else '?'} | Stage: {exc.stage}", flush=True)
    print("!" * 80, flush=True)

    if model is not None and out_dir:
        try:
            os.makedirs(out_dir, exist_ok=True)
            emerg_ckpt = os.path.join(
                out_dir, f"fold{fold}_emergency_{exc.threshold_gb:.0f}gb_checkpoint.pt"
            )
            base_m = model.module if isinstance(model, nn.DataParallel) else model
            _atomic_torch_save(
                {
                    "model": base_m.state_dict(),
                    "cfg": dataclasses.asdict(cfg) if cfg and hasattr(cfg, "__dataclass_fields__") else cfg,
                    "targets": config.TARGETS,
                    "fold": fold,
                    "epoch": epoch,
                    "step": step,
                    "used_gb": exc.used_gb,
                    "variant": variant,
                    "use_cross_slot": use_cross_slot,
                    "model_type": (model_config or {}).get("model_type", "dinov2"),
                    "model_config": model_config or {},
                },
                emerg_ckpt,
            )
            print(f" [CHECKPOINT] Emergency state saved to: {emerg_ckpt}", flush=True)
        except Exception as save_err:
            print(f" [WARNING] Could not write emergency checkpoint: {save_err}", flush=True)

    print(" [FLUSHING] Purging PyTorch CUDA allocations, worker pools, and OS arenas...", flush=True)
    execute_emergency_memory_flush(active_loaders=active_loaders)

    print("\n" + "=" * 80, flush=True)
    print(" CLEAN TERMINATION COMPLETE: HARDWARE PROTECTED FROM HARD LOCKUP / OOM FREEZE.", flush=True)
    print("=" * 80, flush=True)
    print(" PROMPT TO RESTART:", flush=True)
    print("  1. Flush Linux filesystem page cache on DGX:", flush=True)
    print("       sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches", flush=True)
    print(f"  2. Re-launch the training pipeline for Fold {fold}:", flush=True)
    print(f"       python src/main.py --folds {fold} --model_dir <fresh-model-dir>", flush=True)
    print("=" * 80 + "\n", flush=True)


# ─────────────────────────────────────── autocast helper ─────────────────────
def _autocast(device: torch.device):
    if device.type != "cuda":
        return torch.autocast("cpu", enabled=False), False
    use_bf16 = torch.cuda.is_bf16_supported()
    return torch.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16), not use_bf16


# ─────────────────────────────────────────── build LR schedule ────────────────
def _build_schedule(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_frac: float = 0.05,
    eta_min: float = 1e-7,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup (5% steps) + cosine decay to eta_min.

    Source: ViT fine-tuning best practices 2025 (Touvron DeiT III, Oquab DINOv2).
    Replaces OneCycleLR which can overshoot for large ViTs.
    """
    warmup_steps = max(1, int(warmup_frac * total_steps))
    cosine_steps = max(1, total_steps - warmup_steps)
    warmup  = LinearLR(optimizer, start_factor=0.01, total_iters=warmup_steps)
    cosine  = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_steps])


# ──────────────────────────────────────────── Training epoch ─────────────────
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler,
    device: torch.device,
    criterion: nn.Module,
    epoch: int = 1,
    total_epochs: int = 20,
    label_smoothing: float = 0.0,
    grad_accum: int = 1,
    gold_weight_mult: float = 1.0,
) -> float:
    if grad_accum < 1:
        raise ValueError("grad_accum must be at least 1")
    model.train()
    total_loss_tensor = torch.tensor(0.0, device=device)
    n = 0
    # Pre-cache trainable parameters to avoid scanning module tree at every micro-step
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    step_t0 = time.time()
    total_batches = len(dataloader)
    for micro_step, batch in enumerate(dataloader):
        if micro_step == 0:
            step_t0 = time.time()  # Reset timer AFTER the massive 3-minute DataLoader prefetch penalty!
        # ── HARD UNIFIED MEMORY CIRCUIT BREAKER (CHECKED EVERY MICRO-STEP) ──
        check_memory_circuit_breaker(
            stage=f"Training Epoch {epoch}/{total_epochs} Step {micro_step + 1}/{total_batches}"
        )
        imgs, masks, wmasks, targets, weights = batch
        imgs    = imgs.to(device, non_blocking=True)
        masks   = masks.to(device, non_blocking=True)
        wmasks  = wmasks.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)

        ctx, _ = _autocast(device)
        with ctx:
            logits = model(imgs, masks, wmasks)  # [B, C]

        accumulation_steps = min(
            grad_accum,
            total_batches - (micro_step // grad_accum) * grad_accum,
        )
        # Scale only verified gold entries; pseudo-labels in the same study
        # retain their confidence weights.
        if gold_weight_mult > 1.0:
            verified_gold = weights >= 0.99
            weights = weights * (
                1.0 + (gold_weight_mult - 1.0) * verified_gold
            )

        # UPGRADE 6: Curriculum label weighting
        if epoch < 5:
            # Up-weight positive findings
            has_positive = (targets >= 0.5).float()
            weights = weights * (1.0 + has_positive)
            
        # Optional label smoothing
        if label_smoothing > 0.0:
            targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

        with ctx:
            loss = criterion(logits.float(), targets, weights)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        is_last = (micro_step + 1) == len(dataloader)
        if (micro_step + 1) % grad_accum == 0 or is_last:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 0.5, foreach=True)
            scale_before_step = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if not scaler.is_enabled() or scaler.get_scale() >= scale_before_step:
                scheduler.step()

        total_loss_tensor += loss.detach() * accumulation_steps
        n += 1

        if (micro_step + 1) % 10 == 0 or (micro_step + 1) == total_batches:
            dt = time.time() - step_t0
            n_step_win = 10 if (micro_step + 1) % 10 == 0 else max(1, (micro_step + 1) % 10)
            sec_per_step = dt / n_step_win
            rem_sec = (total_batches - (micro_step + 1)) * sec_per_step
            avg_loss = (total_loss_tensor.item()) / max(n, 1)
            sys_mem = psutil.virtual_memory()
            sys_ram_gb = sys_mem.used / (1024**3)
            total_ram_gb = sys_mem.total / (1024**3)
            limit_gb = config.CIRCUIT_BREAKER_MAX_RAM_GB
            headroom_gb = max(0.0, limit_gb - sys_ram_gb)
            available_gb = sys_mem.available / (1024**3)
            vram_alloc = torch.cuda.memory_allocated() / (1024 ** 3) if device.type == "cuda" else 0.0
            vram_res = torch.cuda.memory_reserved() / (1024 ** 3) if device.type == "cuda" else 0.0
            mem_str = (
                f" | VRAM: {vram_alloc:.1f}/{vram_res:.1f}GiB"
                f" | Unified RAM: {sys_ram_gb:.1f}/{total_ram_gb:.0f}GiB"
                f" (available {available_gb:.1f}GiB; target {config.MEMORY_TARGET_GB:.0f}GiB;"
                f" ceiling headroom {headroom_gb:.1f}GiB)"
            )
            print(f"  [Epoch {epoch:2d}/{total_epochs:2d} | Step {micro_step + 1:3d}/{total_batches:3d}] Loss: {avg_loss:.4f} | {sec_per_step:.2f}s/step | ETA: {rem_sec/60:.1f}m{mem_str}", flush=True)
            step_t0 = time.time()

    ret_loss = float(total_loss_tensor.item()) / max(n, 1)
    # UNIFIED MEMORY AUDIT: Purge all batch and intermediate tensors from local scope
    del total_loss_tensor, trainable_params
    if 'batch' in locals():
        del batch, imgs, masks, wmasks, targets, weights
    if 'logits' in locals():
        del logits
    if 'loss' in locals():
        del loss
    return ret_loss


# ─────────────────────────────────────────────── Evaluation ──────────────────
@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[float, dict[str, float]]:
    """Macro AUC over labels that have both classes; weight==0 entries masked."""
    from sklearn.metrics import roc_auc_score
    model.eval()
    P, Y, W = [], [], []
    ctx, _ = _autocast(device)
    total_val_batches = len(dataloader)
    for val_step, batch in enumerate(dataloader):
        check_memory_circuit_breaker(
            stage=f"Validation Evaluation Step {val_step + 1}/{total_val_batches}"
        )
        imgs, masks, wmasks, targets, weights = batch
        with ctx:
            p = model(imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True), wmasks.to(device, non_blocking=True))
        P.append(torch.sigmoid(p.float()).cpu().numpy())
        Y.append(targets.numpy())
        W.append(weights.numpy())
        del imgs, masks, wmasks, targets, weights, p, batch
    P, Y, W = np.concatenate(P), np.concatenate(Y), np.concatenate(W)
    per: dict[str, float] = {}
    for j, t in enumerate(config.TARGETS):
        m = W[:, j] >= 0.49
        y_eval = Y[m, j]
        if m.sum() > 1 and 0 < y_eval.sum() < m.sum():
            per[t] = float(roc_auc_score(y_eval, P[m, j]))
    macro = float(np.mean(list(per.values()))) if per else float("nan")
    del P, Y, W
    return macro, per


# ──────────────────────────────────────────── Main entry point ───────────────
def run_training(
    labels_csv: str,
    cache_prefix: str,
    folds_csv: str | None = None,
    fold: int = 0,
    out_dir: str = ".",
    epochs: int | None = None,
    n_windows_train: int = config.N_WINDOWS_TRAIN,
    num_workers: int = config.NUM_WORKERS,
    prefetch_factor: int = config.PREFETCH_FACTOR,
    eval_batch_size: int = config.EVAL_BATCH_SIZE,
    seed: int = config.SEED,
    variant: str = "dinov2-base",         # Upgrade A: default to Base
    model_type: str | None = None,
    model_input_size: int = config.COATNET_INPUT_SIZE,
    encode_chunk_size: int = config.COATNET_ENCODE_CHUNK,
    pretrained: bool = True,
    unfreeze_last: int = config.UNFREEZE_LAST,
    use_cross_slot: bool = True,          # Upgrade C
    lora_rank: int = config.LORA_RANK,    # Upgrade B
    lora_alpha: int = config.LORA_ALPHA,
    # loss
    loss_name: str = "asl",              # Upgrade B: ASL by default
    asl_gamma_neg: float = config.ASL_GAMMA_NEG,
    asl_gamma_pos: float = config.ASL_GAMMA_POS,
    asl_clip: float = config.ASL_CLIP,
    label_smoothing: float = config.LABEL_SMOOTHING,
    # training recipe
    batch_size: int = config.BATCH_SIZE,
    grad_accum: int = config.GRAD_ACCUM,
    swa_epochs: int = config.SWA_EPOCHS,       # Upgrade E
    early_stop_patience: int = config.EARLY_STOP_PATIENCE,
    timm_pooling: str = "hierarchical",
    use_slot_prior: bool = True,
) -> float:
    """Train one fold and save the best checkpoint + optional SWA checkpoint.

    For 5-fold CV (Upgrade G): run once per fold from separate notebook sessions
    and pass folds_csv pointing to a patient-stratified fold CSV.  Final test
    predictions = average of all fold model outputs (before rank ensemble).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg    = cache_cfg(cache_prefix)   # config the cache was BUILT with — never re-derive
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df     = pd.read_csv(labels_csv)
    model_type = model_type or ("coatnet_mil" if variant.startswith("coatnet") else ("timm_mil" if not variant.startswith("dinov2") else "dinov2"))
    if model_type not in {"dinov2", "coatnet_mil", "timm_mil"}:
        raise ValueError(f"Unsupported model_type: {model_type}")

    if folds_csv:
        fold_table = pd.read_csv(
            folds_csv, dtype={"StudyInstanceUID": str}
        )[["StudyInstanceUID", "fold"]]
        if fold_table["StudyInstanceUID"].duplicated().any():
            raise ValueError("folds manifest contains duplicate StudyInstanceUID values")
        if set(fold_table["StudyInstanceUID"]) != set(df["StudyInstanceUID"].astype(str)):
            raise ValueError("folds manifest must cover the label table exactly")
        fold_values = pd.to_numeric(fold_table["fold"], errors="raise")
        if (
            fold_values.isna().any()
            or not np.equal(fold_values, fold_values.astype(int)).all()
            or not set(fold_values.astype(int)).issubset(set(range(5)))
        ):
            raise ValueError("folds manifest must assign integer fold IDs in 0..4")
        fold_table["fold"] = fold_values.astype(int)
        df["StudyInstanceUID"] = df["StudyInstanceUID"].astype(str)
        df = df.merge(
            fold_table,
            on="StudyInstanceUID",
            how="inner",
            validate="one_to_one",
        )
        tr, va = df[df.fold != fold], df[df.fold == fold]
    else:
        tr, va = df, df.iloc[:0]

    n_ep = config.EPOCHS if epochs is None else epochs
    if not 0 <= fold < 5:
        raise ValueError(f"fold must be in 0..4, got {fold}")
    if n_ep < 1 or n_windows_train < 1 or early_stop_patience < 1:
        raise ValueError(
            "epochs, n_windows_train, and early_stop_patience must be positive"
        )
    if swa_epochs < 0 or swa_epochs > n_ep:
        raise ValueError(f"swa_epochs must be in 0..{n_ep}, got {swa_epochs}")
    if label_smoothing < 0 or label_smoothing >= 1:
        raise ValueError("label_smoothing must be in [0, 1)")
    if (
        batch_size < 1
        or grad_accum < 1
        or num_workers < 0
        or prefetch_factor < 1
        or eval_batch_size < 1
    ):
        raise ValueError(
            "batch_size, grad_accum, prefetch_factor, and eval_batch_size must be "
            ">= 1; num_workers must be >= 0"
        )
    if tr.empty:
        raise ValueError(f"fold {fold} has no training studies")
    print(
        f"device={device}  train={len(tr)}  val={len(va)}  fold={fold}  "
        f"preset={cfg.name}  D={cfg.stack_depth}  img={cfg.img_size}  "
        f"model={model_type}/{variant}  epochs={n_ep}  loss={loss_name}  "
        f"swa={swa_epochs}ep"
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(
        variant=variant,
        unfreeze_last=unfreeze_last,
        use_cross_slot=use_cross_slot,
        use_slot_prior=use_slot_prior,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        model_type=model_type,
        input_size=model_input_size,
        encode_chunk_size=encode_chunk_size,
        timm_pooling=timm_pooling,
        pretrained=pretrained,
    ).to(device)
    if model_type in ("coatnet_mil", "timm_mil"):
        model_config = {
            "model_type": model_type,
            "variant": variant,
            "input_size": model_input_size,
            "encode_chunk_size": encode_chunk_size,
            "timm_pooling": timm_pooling,
            "pretrained": False,
        }
    else:
        model_config = {
            "model_type": model_type,
            "variant": variant,
            "unfreeze_last": unfreeze_last,
            "use_cross_slot": use_cross_slot,
            "use_slot_prior": use_slot_prior,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "truncate_blocks": 0,
        }
        model_config["backbone_config"] = model.backbone.config.to_dict()

    # Multi-GPU DataParallel for extreme throughput on DGX systems
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if num_gpus > 1:
        print(f"  [HARDWARE] DGX Multi-GPU detected: Enabling DataParallel across {num_gpus} GPUs!", flush=True)
        model_train = nn.DataParallel(model)
    else:
        model_train = model

    # ── Loss ─────────────────────────────────────────────────────────────────
    criterion = build_loss(
        loss_name,
        **({"gamma_neg": asl_gamma_neg, "gamma_pos": asl_gamma_pos, "clip": asl_clip}
           if loss_name == "asl" else {}),
    ).to(device)

    # ── Datasets ─────────────────────────────────────────────────────────────
    num_workers = min(num_workers, os.cpu_count() or 1)
    gold_mask = (
        va[[f"{target}_weight" for target in config.TARGETS]]
        .fillna(0.0)
        .ge(0.99)
        .any(axis=1)
    ) if len(va) else pd.Series(dtype=bool)
    ds_tr = RSNADataset(tr, cache_prefix, cfg, True, n_windows_train, seed)
    ds_va = RSNADataset(va, cache_prefix, cfg, False, None, seed, aug=False) if len(va) else None
    ds_oof = RSNADataset(va, cache_prefix, cfg, False, None, seed, aug=False) if len(va) else None
    if len(ds_tr) != len(tr):
        raise RuntimeError(
            f"training cache is missing {len(tr) - len(ds_tr)} studies from fold {fold}"
        )
    if ds_oof is not None and len(ds_oof) != len(va):
        raise RuntimeError(
            f"training cache is missing {len(va) - len(ds_oof)} held-out studies "
            f"from fold {fold}"
        )
    queued_train_gb = (
        _loader_batch_bytes(cfg, n_windows_train, batch_size)
        * num_workers * prefetch_factor / (1024 ** 3)
    )
    print(
        f"loader workers={num_workers}, prefetch={prefetch_factor}, "
        f"estimated prefetched image payload≈{queued_train_gb:.1f} GiB; "
        f"validation batch={eval_batch_size}, gold studies={len(va_gold)}/{len(va)}; "
        f"memory target={config.MEMORY_TARGET_GB:.0f} GiB, "
        f"hard ceiling={config.CIRCUIT_BREAKER_MAX_RAM_GB:.0f} GiB"
    )

    # ── Optimiser (dual LR: slow backbone, fast head) ────────────────────────
    # IMPROVEMENT 3: WD=0 for biases and norm layers (standard ViT recipe).
    # DINOv2 Pre-LayerNorm scale params must not be decayed.
    def _is_no_decay(name):
        components = name.lower().split(".")
        return name.endswith(".bias") or any(
            component.startswith(("norm", "bn")) for component in components
        )

    # FIX: inner_model was never defined; the variable is just `model`.
    inner_model = model  # alias for clarity in checkpoint saving below
    backbone_params = []
    head_params = []
    if model_type in ("coatnet_mil", "timm_mil"):
        named_groups = [
            (
                [(f"backbone.{name}", parameter)
                 for name, parameter in inner_model.backbone.named_parameters()],
                config.COATNET_LR_BACKBONE,
                backbone_params,
            ),
            (
                [(name, parameter) for name, parameter in inner_model.named_parameters()
                 if not name.startswith("backbone.")],
                config.COATNET_LR_HEAD,
                head_params,
            ),
        ]
        for named_parameters, learning_rate, groups in named_groups:
            for no_decay in (False, True):
                parameters = [
                    parameter for name, parameter in named_parameters
                    if parameter.requires_grad and _is_no_decay(name) == no_decay
                ]
                if parameters:
                    groups.append({
                        "params": parameters,
                        "lr": learning_rate,
                        "weight_decay": 0.0 if no_decay else config.COATNET_WEIGHT_DECAY,
                    })
    else:
        # Layer-wise learning-rate decay for the transformer backbone.
        llrd_decay = 0.85
        n_layer = len(inner_model.backbone.encoder.layer)
        for i, block in enumerate(inner_model.backbone.encoder.layer):
            layer_lr = config.LR_BACKBONE * (llrd_decay ** (n_layer - i))
            wd_params = [
                p for n, p in block.named_parameters()
                if p.requires_grad and not _is_no_decay(n)
            ]
            no_wd_params = [
                p for n, p in block.named_parameters()
                if p.requires_grad and _is_no_decay(n)
            ]
            if wd_params:
                backbone_params.append({
                    "params": wd_params, "lr": layer_lr,
                    "weight_decay": config.WEIGHT_DECAY,
                })
            if no_wd_params:
                backbone_params.append({
                    "params": no_wd_params, "lr": layer_lr, "weight_decay": 0.0,
                })
        embed_lr = config.LR_BACKBONE * (llrd_decay ** n_layer)
        embed_wd = [
            p for n, p in inner_model.backbone.named_parameters()
            if p.requires_grad and "encoder.layer" not in n and not _is_no_decay(n)
        ]
        embed_no_wd = [
            p for n, p in inner_model.backbone.named_parameters()
            if p.requires_grad and "encoder.layer" not in n and _is_no_decay(n)
        ]
        if embed_wd:
            backbone_params.append({
                "params": embed_wd, "lr": embed_lr,
                "weight_decay": config.WEIGHT_DECAY,
            })
        if embed_no_wd:
            backbone_params.append({
                "params": embed_no_wd, "lr": embed_lr, "weight_decay": 0.0,
            })
        head_params = (
            list(inner_model.wpool.parameters())
            + list(inner_model.head.parameters())
            + (list(inner_model.cross_slot.parameters()) if inner_model.use_cross_slot else [])
        )
    optimizer_groups = (
        head_params + backbone_params
        if model_type in ("coatnet_mil", "timm_mil")
        else [{"params": head_params, "lr": config.LR_HEAD, "weight_decay": 0.0}]
        + backbone_params
    )
    optimizer = torch.optim.AdamW(optimizer_groups, fused=(device.type == "cuda"))
    optimizer_parameter_ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    trainable_parameter_ids = {
        id(parameter)
        for parameter in inner_model.parameters()
        if parameter.requires_grad
    }
    if (
        len(optimizer_parameter_ids) != len(set(optimizer_parameter_ids))
        or set(optimizer_parameter_ids) != trainable_parameter_ids
    ):
        raise RuntimeError(
            "Optimizer parameter groups must contain every trainable parameter "
            "exactly once"
        )

    # ?? LR Schedule (Upgrade J: cosine with warmup) ??????????????????????????
    train_batches = (
        len(tr) // batch_size
        if len(tr) > batch_size
        else int(len(tr) > 0)
    )
    steps_per_epoch = math.ceil(train_batches / grad_accum) if train_batches else 1
    total_steps = n_ep * max(steps_per_epoch, 1)
    scheduler = _build_schedule(optimizer, total_steps, warmup_frac=0.08)

    # ── Mixed precision ───────────────────────────────────────────────────────
    _, need_scaler = _autocast(device)
    scaler = torch.amp.GradScaler("cuda", enabled=need_scaler)

    # ── SWA model (Upgrade E, Izmailov et al. UAI 2018) ──────────────────────
    swa_start = max(0, n_ep - swa_epochs)
    swa_model: AveragedModel | None = None
    if swa_epochs > 0:
        swa_model = AveragedModel(model)

    os.makedirs(out_dir, exist_ok=True)
    best, t0 = -1.0, time.time()
    epochs_no_improve = 0

    dl_tr = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=True,
        drop_last=len(tr) > batch_size, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    dl_va = (
        DataLoader(ds_va, batch_size=eval_batch_size, shuffle=False,
                   num_workers=0, pin_memory=False)
        if ds_va is not None else None
    )
    active_loaders = [dl_tr]
    try:
        for ep in range(n_ep):
            ds_tr.set_epoch(ep)
            
            # UPGRADE 4: Annealed attention temperature
            cur_temp = max(0.3, 1.0 - 0.7 * (ep / max(1, n_ep)))
            if hasattr(model, 'module'):
                if hasattr(model.module, 'wpool'):
                    model.module.wpool.temperature = cur_temp
            elif hasattr(model, 'wpool'):
                model.wpool.temperature = cur_temp

            # Gold upweighting is a fixed heuristic and has not been isolated
            # in a matched validation ablation.
            loss_val = train_epoch(
                model_train, dl_tr, optimizer, scaler, scheduler, device,
                criterion,
                epoch=ep + 1,
                total_epochs=n_ep,
                label_smoothing=label_smoothing,
                grad_accum=grad_accum,
                gold_weight_mult=2.0,
            )

            msg = f"epoch {ep + 1}/{n_ep}  loss {loss_val:.4f}"
            score = float("nan")

            if dl_va is not None:
                score, per = evaluate(model, dl_va, device)
                msg += f"  val macro-AUC {score:.4f}"
                all_conds = ", ".join(f"{k}:{v:.3f}" for k, v in sorted(per.items()))
                msg += f"\n       Per-target AUCs ({len(per)}/12 evaluated): [{all_conds}]"

            print(msg, flush=True)

            # ── SWA weight accumulation ───────────────────────────────────────
            if swa_model is not None and ep >= swa_start:
                swa_model.update_parameters(model)

            # ── Best-checkpoint saving ────────────────────────────────────────
            save = True
            if save:
                best = score
                epochs_no_improve = 0
                ckpt_file = os.path.join(out_dir, f"fold{fold}_last.pt")
                _atomic_torch_save(
                    dict(
                        model=inner_model.state_dict(),
                        cfg=dataclasses.asdict(cfg),
                        model_config=model_config,
                        targets=config.TARGETS,
                        fold=fold,
                        epoch=ep + 1,
                        val=score,
                        variant=variant,
                        use_cross_slot=use_cross_slot,
                    ),
                    ckpt_file,
                )
                print(f"  [CHECKPOINT] New best validation AUC: {best:.4f} -> Saved {ckpt_file}", flush=True)
            else:
                epochs_no_improve += 1
                print(f"  [EARLY STOP] No validation AUC improvement for {epochs_no_improve}/{early_stop_patience} epochs (best: {best:.4f})", flush=True)
                if ds_va is not None and epochs_no_improve >= early_stop_patience:
                    print(f"  [EARLY STOP] Validation AUC did not improve for {early_stop_patience} consecutive epochs. Stopping early at epoch {ep + 1}/{n_ep} to prevent overfitting.", flush=True)
                    break

            # ── Time-budget guard ─────────────────────────────────────────────
            if time.time() - t0 > config.TIME_BUDGET_HOURS * 3600:
                print("Time budget reached; stopping early.")
                break

        # ── SWA: recompute BatchNorm stats, save ─────────────────────────────
        def safe_update_bn(loader, model, device):
            momenta = {}
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.reset_running_stats()
                    momenta[module] = module.momentum
            if not momenta:
                return
            was_training = model.training
            model.train()
            for module in momenta:
                module.momentum = None
            total_bn_batches = len(loader)
            for bn_step, batch in enumerate(loader):
                check_memory_circuit_breaker(
                    stage=f"SWA BatchNorm Step {bn_step + 1}/{total_bn_batches}"
                )
                if isinstance(batch, (list, tuple)):
                    imgs = batch[0].to(device)
                    mask = batch[1].to(device) if len(batch) > 1 else None
                    wmask = batch[2].to(device) if len(batch) > 2 else None
                    model(imgs, mask, wmask)
                    del imgs, mask, wmask
                else:
                    imgs = batch.to(device)
                    model(imgs)
                    del imgs
                del batch
            for bn_module in momenta:
                bn_module.momentum = momenta[bn_module]
            model.train(was_training)

        if swa_model is not None and swa_epochs > 0 and (ep >= swa_start):
            print("Updating SWA batch-norm statistics …")
            dl_swa = DataLoader(
                ds_tr, batch_size=batch_size, shuffle=True,
                drop_last=len(tr) > batch_size, num_workers=min(4, num_workers),
                pin_memory=(device.type == "cuda"),
                persistent_workers=False,
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            )
            active_loaders = [dl_swa]
            safe_update_bn(dl_swa, swa_model, device=device)
            active_loaders = []
            del dl_swa
            _atomic_torch_save(
                dict(
                    model=swa_model.module.state_dict(),
                    cfg=dataclasses.asdict(cfg),
                    model_config=model_config,
                    targets=config.TARGETS,
                    fold=fold,
                    epoch=ep + 1,
                    val=best,
                    swa=True,
                    variant=variant,
                    use_cross_slot=use_cross_slot,
                ),
                os.path.join(out_dir, f"fold{fold}_swa.pt"),
            )
            print(f"SWA checkpoint saved → fold{fold}_swa.pt")

        _shutdown_loader(dl_tr)
        active_loaders = []
        if ds_oof is not None:
            best_path = os.path.join(out_dir, f"fold{fold}_last.pt")
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
            inner_model.load_state_dict(best_checkpoint["model"], strict=True)
            inner_model.eval()
            dl_oof = DataLoader(
                ds_oof,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=min(4, num_workers),
                pin_memory=(device.type == "cuda"),
                persistent_workers=(num_workers > 0),
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            )
            active_loaders = [dl_oof]
            oof_probabilities, oof_targets, oof_weights = [], [], []
            ctx, _ = _autocast(device)
            with torch.no_grad():
                for oof_step, batch in enumerate(dl_oof):
                    check_memory_circuit_breaker(
                        stage=f"OOF Prediction Step {oof_step + 1}/{len(dl_oof)}"
                    )
                    imgs, masks, wmasks, targets, weights = batch
                    with ctx:
                        logits = inner_model(
                            imgs.to(device, non_blocking=True),
                            masks.to(device, non_blocking=True),
                            wmasks.to(device, non_blocking=True),
                        )
                    oof_probabilities.append(torch.sigmoid(logits.float()).cpu().numpy())
                    oof_targets.append(targets.numpy())
                    oof_weights.append(weights.numpy())
                    del imgs, masks, wmasks, targets, weights, logits, batch

            oof = pd.DataFrame({"StudyInstanceUID": ds_oof.ids})
            probabilities = np.concatenate(oof_probabilities)
            targets = np.concatenate(oof_targets)
            weights = np.concatenate(oof_weights)
            if len(oof) != len(probabilities):
                raise RuntimeError(
                    f"fold {fold}: OOF identifier/prediction count mismatch "
                    f"({len(oof)} != {len(probabilities)})"
                )
            for j, target in enumerate(config.TARGETS):
                oof[f"pred_{target}"] = probabilities[:, j]
                oof[f"target_{target}"] = targets[:, j]
                oof[f"weight_{target}"] = weights[:, j]
            oof_path = os.path.join(out_dir, f"fold{fold}_oof.csv")
            oof_tmp = f"{oof_path}.tmp"
            oof.to_csv(oof_tmp, index=False)
            os.replace(oof_tmp, oof_path)
            print(f"OOF predictions from the selected best checkpoint saved → {oof_path}")
            _shutdown_loader(dl_oof)
            active_loaders = []
            del best_checkpoint, dl_oof, oof_probabilities, oof_targets, oof_weights
            del probabilities, targets, weights, oof

    except MemoryCircuitBreakerTriggered as exc:
        handle_memory_circuit_breaker(
            exc=exc,
            fold=fold,
            epoch=ep + 1 if 'ep' in locals() else None,
            step=None,
            model=inner_model if 'inner_model' in locals() else None,
            cfg=cfg if 'cfg' in locals() else None,
            variant=variant,
            use_cross_slot=use_cross_slot,
            model_config=model_config if "model_config" in locals() else None,
            out_dir=out_dir,
            active_loaders=active_loaders,
        )
        raise exc
    except Exception:
        execute_emergency_memory_flush(active_loaders=active_loaders)
        raise

    # ── Final cleanup before exiting fold training ───────────────────────────
    _shutdown_loader(dl_tr)
    del dl_tr
    if dl_va is not None:
        del dl_va
    del model, inner_model
    if 'model_train' in locals():
        del model_train
    del optimizer, scheduler, scaler, ds_tr
    if ds_va is not None:
        del ds_va
    if swa_model is not None:
        del swa_model
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return best
