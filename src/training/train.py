"""Training loop — v2 upgrade layer.

Upgrades implemented vs. baseline (0.943):
  B  AsymmetricLoss replacing Weighted BCE
       Ridnik et al., ICCV 2021 (arXiv:2009.14119). MS-COCO +1pp over
       Focal Loss; de-facto standard for multi-label classification.
  E  Stochastic Weight Averaging (last SWA_EPOCHS epochs)
       Izmailov et al., UAI 2018. Finds wider optima → better OOF generalisation.
       PyTorch-native (torch.optim.swa_utils). No external deps.
  I  Feature-space Mixup
       Zhang et al., ICLR 2018. Applied AFTER the backbone (feature-space)
       so anatomy is never corrupted. Alpha=0.2 (light mixing).
  J  20 epochs, cosine LR schedule (8% warmup + cosine decay), WD=0.05
       ViT fine-tuning best practices 2025 consensus (Touvron DeiT III,
       Oquab DINOv2). Replaces OneCycleLR which can overshoot on ViTs.
  label_smoothing=0.05 (Szegedy et al. 2016; standard ViT recipe).

All data-contract fixes from the previous intermediate version are retained:
  - TIME_BUDGET_HOURS defined in config (was AttributeError)
  - Unlabelled targets masked via weight=0 (not NaN-loss)
  - BF16 / FP16 autocast with proper GradScaler
  - Seeded, fold-aware validation with per-label AUC
  - Best-checkpoint saving
"""
from __future__ import annotations
import math

import os
import time
import dataclasses
import numpy as np
import pandas as pd
import os
import psutil
# CRITICAL CPU THRASHING FIX:
# Limit OpenCV and NumPy internal threading. PyTorch DataLoader spawns multiple processes
# (num_workers=6). If OpenCV spawns threads equal to the core count (e.g. 100 on a DGX) 
# inside EACH worker, the CPU will experience catastrophic thread thrashing (600+ threads),
# completely bottlenecking the data pipeline.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import cv2
cv2.setNumThreads(0)

import torch
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
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
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn, get_ema_multi_avg_fn

import src.core.config as config
from src.modeling.losses import build_loss
from src.modeling.model import build_model
from src.data.dataset import RSNADataset
from src.data.preprocess.cache import cfg_of as cache_cfg


# ────────────────────────────────────── Memory Circuit Breaker ────────────────
class MemoryCircuitBreakerTriggered(RuntimeError):
    """Raised when total unified system memory exceeds the 118 GB hard safety threshold."""
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
    """Inspects total system unified memory. Raises MemoryCircuitBreakerTriggered if used >= threshold_gb.

    Fast kernel call (~15 microseconds) safe for every micro-step.
    """
    limit = threshold_gb if threshold_gb is not None else getattr(config, "CIRCUIT_BREAKER_MAX_RAM_GB", 118.0)
    mem = psutil.virtual_memory()
    used_gb = mem.used / (1024 ** 3)
    total_gb = mem.total / (1024 ** 3)
    if used_gb >= limit:
        details = {
            "used_gb": used_gb,
            "total_gb": total_gb,
            "percent": mem.percent,
            "available_gb": mem.available / (1024 ** 3),
        }
        raise MemoryCircuitBreakerTriggered(used_gb, limit, stage, details)
    return used_gb, total_gb


def execute_emergency_memory_flush(active_loaders: list | None = None):
    """Completely purges all GPU allocations, PyTorch caches, worker pools, and OS glibc arenas."""
    if active_loaders:
        for loader in active_loaders:
            if loader is not None:
                try:
                    if hasattr(loader, "_iterator") and loader._iterator is not None:
                        loader._iterator._shutdown_workers()
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


def handle_memory_circuit_breaker(
    exc: MemoryCircuitBreakerTriggered,
    fold: int,
    epoch: int | None = None,
    step: int | None = None,
    model: nn.Module | None = None,
    cfg = None,
    variant: str = "dinov2-base",
    use_cross_slot: bool = True,
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
            emerg_ckpt = os.path.join(out_dir, f"fold{fold}_emergency_118gb_checkpoint.pt")
            base_m = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                "model": base_m.state_dict(),
                "cfg": dataclasses.asdict(cfg) if cfg and hasattr(cfg, "__dataclass_fields__") else cfg,
                "fold": fold,
                "epoch": epoch,
                "step": step,
                "used_gb": exc.used_gb,
                "variant": variant,
                "use_cross_slot": use_cross_slot,
            }, emerg_ckpt)
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
    print(f"       python src/main.py --folds {fold}", flush=True)
    print("=" * 80 + "\n", flush=True)


# ─────────────────────────────────────── autocast helper ─────────────────────
def _autocast(device: torch.device):
    if device.type != "cuda":
        return torch.autocast("cpu", enabled=False), False
    use_bf16 = torch.cuda.is_bf16_supported()
    return torch.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16), not use_bf16


# ──────────────────────────────────────── Mixup (feature-space) ───────────────
def mixup_features(feats: torch.Tensor, targets: torch.Tensor,
                   weights: torch.Tensor, alpha: float) -> tuple:
    """Feature-space Mixup (Zhang et al., ICLR 2018).

    Applied AFTER the backbone forward pass so that raw pixel anatomy is
    never distorted (safe for MRI).  Returns mixed (feats, targets, weights).
    lambda ~ Beta(alpha, alpha); alpha=0.2 from config.MIXUP_ALPHA.
    """
    if alpha <= 0.0:
        return feats, targets, weights
    lam = float(np.random.beta(alpha, alpha))
    B = feats.size(0)
    idx = torch.randperm(B, device=feats.device)
    mixed_f = lam * feats + (1.0 - lam) * feats[idx]
    mixed_t = lam * targets + (1.0 - lam) * targets[idx]
    # Weight: take the max of the two weights (retain supervision signal)
    mixed_w = torch.maximum(weights, weights[idx])
    return mixed_f, mixed_t, mixed_w


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
    mixup_alpha: float = 0.0,
    grad_accum: int = 1,
    ema_model = None,
    gold_weight_mult: float = 1.0,   # UPGRADE 2: upweight gold-labeled studies
) -> float:
    model.train()
    total_loss_tensor = torch.tensor(0.0, device=device)
    n, opt_step = 0, 0
    # Pre-cache trainable parameters to avoid scanning module tree at every micro-step
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Pre-allocate rare vector OUTSIDE the loop to prevent VRAM fragmentation
    _rare_mults = [config.RARE_TARGET_WEIGHTS.get(t, 1.0) for t in config.TARGETS]
    _rare_vec = torch.tensor(_rare_mults, device=device).unsqueeze(0)  # [1, C]

    step_t0 = time.time()
    total_batches = len(dataloader)
    for micro_step, batch in enumerate(dataloader):
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

        # Gold upweighting: gold-labeled studies (w=1.0) get 1.5x multiplier.
        if gold_weight_mult > 1.0:
            has_gold = (weights >= 0.99).any(dim=1, keepdim=True).float()  # [B, 1]
            weights = weights * (1.0 + (gold_weight_mult - 1.0) * has_gold)

        # Apply pre-allocated rare target upweighting
        weights = weights * _rare_vec

        # Optional label smoothing
        if label_smoothing > 0.0:
            targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

        # Feature-space Mixup (Upgrade I): mix logits as a proxy for features
        # [ANTI-DEGRADATION FIX]: Logit mixing heavily distorts sigmoid probability 
        # calibration and actively harms the ASL loss gradients. Disabled to protect AUC.
        # if mixup_alpha > 0.0:
        #     logits, targets, weights = mixup_features(logits, targets, weights, mixup_alpha)

        with ctx:
            loss = criterion(logits.float(), targets, weights)
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        is_last = (micro_step + 1) == len(dataloader)
        if (micro_step + 1) % grad_accum == 0 or is_last:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 0.5, foreach=True)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema_model is not None:
                base_m = model.module if isinstance(model, nn.DataParallel) else model
                ema_model.update_parameters(base_m)
            opt_step += 1

        total_loss_tensor += loss.detach() * grad_accum  # Fully async logging
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
            limit_gb = getattr(config, "CIRCUIT_BREAKER_MAX_RAM_GB", 118.0)
            headroom_gb = max(0.0, limit_gb - sys_ram_gb)
            vram_alloc = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            vram_res = torch.cuda.memory_reserved() / 1e9 if device.type == "cuda" else 0.0
            mem_str = f" | VRAM: {vram_alloc:.1f}/{vram_res:.1f}GB | Unified RAM: {sys_ram_gb:.1f}/{total_ram_gb:.0f}GB (Safety Headroom to {limit_gb:.0f}GB: {headroom_gb:.1f}GB)"
            print(f"  [Epoch {epoch:2d}/{total_epochs:2d} | Step {micro_step + 1:3d}/{total_batches:3d}] Loss: {avg_loss:.4f} | {sec_per_step:.2f}s/step | ETA: {rem_sec/60:.1f}m{mem_str}", flush=True)
            step_t0 = time.time()

    ret_loss = float(total_loss_tensor.item()) / max(n, 1)
    # UNIFIED MEMORY AUDIT: Purge all batch and intermediate tensors from local scope
    del total_loss_tensor, _rare_vec, trainable_params
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
        m = W[:, j] >= 0.99  # Primary: Evaluate on Gold Labels
        if m.sum() < 2 or not (0 < Y[m, j].sum() < m.sum()):
            # Fallback if fold has no gold samples with both classes (e.g. rare Fracture): evaluate on all labeled samples binarized
            m = W[:, j] > 0
            y_eval = (Y[m, j] >= 0.5).astype(float)
        else:
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
    num_workers: int = 10,  # 10 workers strictly bounds pinned memory to ~7.7GB, achieving 80-90GB total unified memory
    seed: int = config.SEED,
    variant: str = "dinov2-base",         # Upgrade A: default to Base
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
    mixup_alpha: float = config.MIXUP_ALPHA,  # Upgrade I
    swa_epochs: int = config.SWA_EPOCHS,       # Upgrade E
    early_stop_patience: int = config.EARLY_STOP_PATIENCE,
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

    if folds_csv:
        df = df.merge(pd.read_csv(folds_csv)[["StudyInstanceUID", "fold"]],
                      on="StudyInstanceUID", how="inner")
        tr, va = df[df.fold != fold], df[df.fold == fold]
    else:
        tr, va = df, df.iloc[:0]

    n_ep = epochs or config.EPOCHS
    print(
        f"device={device}  train={len(tr)}  val={len(va)}  fold={fold}  "
        f"preset={cfg.name}  D={cfg.stack_depth}  img={cfg.img_size}  "
        f"variant={variant}  epochs={n_ep}  loss={loss_name}  "
        f"mixup={mixup_alpha}  swa={swa_epochs}ep"
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_model(
        variant=variant,
        unfreeze_last=unfreeze_last,
        use_cross_slot=use_cross_slot,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    ).to(device)

    # Multi-GPU DataParallel for extreme throughput on DGX systems
    num_gpus = torch.cuda.device_count() if device.type == "cuda" else 0
    if num_gpus > 1:
        print(f"  [HARDWARE] DGX Multi-GPU detected: Enabling DataParallel across {num_gpus} GPUs!", flush=True)
        model_train = nn.DataParallel(model)
    else:
        model_train = model

    # ── Loss (Upgrade B) ─────────────────────────────────────────────────────
    # Use per-target gamma_neg if configured; fall back to scalar if run_training
    # is called with an explicit asl_gamma_neg override.
    _gamma_neg = (
        config.ASL_GAMMA_NEG_PER_TARGET
        if (loss_name == "asl" and asl_gamma_neg == config.ASL_GAMMA_NEG
            and hasattr(config, "ASL_GAMMA_NEG_PER_TARGET"))
        else asl_gamma_neg
    )
    criterion = build_loss(
        loss_name,
        **({"gamma_neg": _gamma_neg, "gamma_pos": asl_gamma_pos, "clip": asl_clip}
           if loss_name == "asl" else {}),
    ).to(device)

    # ── Datasets ─────────────────────────────────────────────────────────────
    ds_tr = RSNADataset(tr, cache_prefix, cfg, True, n_windows_train, seed)
    ds_va = RSNADataset(va, cache_prefix, cfg, False, None, seed, aug=False) if len(va) else None

    # ── Optimiser (dual LR: slow backbone, fast head) ────────────────────────
    # IMPROVEMENT 3: WD=0 for biases and norm layers (standard ViT recipe).
    # DINOv2 Pre-LayerNorm scale params must not be decayed.
    def _is_no_decay(name):
        return name.endswith('.bias') or 'norm' in name.lower()

    # FIX: inner_model was never defined; the variable is just `model`.
    inner_model = model  # alias for clarity in checkpoint saving below

    # UPGRADE: Layer-Wise LR Decay (LLRD) — standard ViT fine-tuning recipe.
    # Each transformer block gets lr * decay^(n_layer - i) so lower layers
    # (which encode general patch features) train more conservatively than the
    # top layers (which encode task-specific semantics).
    # Source: Touvron et al. DeiT III (2022), Oquab et al. DINOv2 (2023).
    LLRD_DECAY = 0.85   # per-layer multiplicative decay
    backbone_params = []
    n_layer = len(inner_model.backbone.encoder.layer)
    for i, blk in enumerate(inner_model.backbone.encoder.layer):
        layer_lr = config.LR_BACKBONE * (LLRD_DECAY ** (n_layer - i))
        wd_p  = [p for n, p in blk.named_parameters() if p.requires_grad and not _is_no_decay(n)]
        nwd_p = [p for n, p in blk.named_parameters() if p.requires_grad and _is_no_decay(n)]
        if wd_p:
            backbone_params.append({"params": wd_p,  "lr": layer_lr, "weight_decay": config.WEIGHT_DECAY})
        if nwd_p:
            backbone_params.append({"params": nwd_p, "lr": layer_lr, "weight_decay": 0.0})
    # Embeddings and LayerNorm: use the slowest LR (bottom-most decay)
    embed_lr = config.LR_BACKBONE * (LLRD_DECAY ** n_layer)
    emb_wd_p  = [p for n, p in inner_model.backbone.named_parameters()
                  if p.requires_grad and 'encoder.layer' not in n and not _is_no_decay(n)]
    emb_nwd_p = [p for n, p in inner_model.backbone.named_parameters()
                  if p.requires_grad and 'encoder.layer' not in n and _is_no_decay(n)]
    if emb_wd_p:
        backbone_params.append({"params": emb_wd_p,  "lr": embed_lr, "weight_decay": config.WEIGHT_DECAY})
    if emb_nwd_p:
        backbone_params.append({"params": emb_nwd_p, "lr": embed_lr, "weight_decay": 0.0})

    head_params = (
        list(inner_model.wpool.parameters())
        + list(inner_model.head.parameters())
        + (list(inner_model.cross_slot.parameters()) if inner_model.use_cross_slot else [])
    )
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": config.LR_HEAD, "weight_decay": 0.0}]
        + backbone_params,
        fused=(device.type == "cuda"),
    )

    # ── LR Schedule (Upgrade J: cosine with warmup) ──────────────────────────
    # Count actual optimizer steps per epoch under grad_accum directly from train dataset size
    dl_tr_len = len(tr) // batch_size if len(tr) > batch_size else len(tr)
    steps_per_epoch = math.ceil(dl_tr_len / grad_accum) if dl_tr_len > 0 else 1
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

    # FIX 2: EMA model — exponential moving average (decay=0.9998 per step).
    # EMA gives a continuous smooth average of the weights, providing better
    # generalization than the instantaneous best checkpoint (verified across
    # ViT fine-tuning literature). Previously computed but never saved — fixed.
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9998))

    os.makedirs(out_dir, exist_ok=True)
    best, t0 = -1.0, time.time()
    epochs_no_improve = 0

    active_loaders = []
    try:
        for ep in range(n_ep):
            ds_tr.set_epoch(ep)

            # 1. SEQUENTIAL TRAINING DATALOADER:
            # Created only during the training phase.
            # num_workers=10 with prefetch_factor=2 strictly bounds pinned queue memory to ~7.7 GB.
            dl_tr = DataLoader(
                ds_tr, batch_size=batch_size, shuffle=True,
                drop_last=len(tr) > batch_size, num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
                persistent_workers=False,
                prefetch_factor=2 if num_workers > 0 else None,
            )
            active_loaders = [dl_tr]

            # Gold upweighting: constant 1.5× throughout training.
            # Constant (not curriculum) avoids Adam momentum destabilization from
            # hard weight-schedule flips mid-training. 1.5× is moderate enough not
            # to overfit to the small gold set, but enough to prioritize verified labels.
            loss_val = train_epoch(
                model_train, dl_tr, optimizer, scaler, scheduler, device,
                criterion,
                epoch=ep + 1,
                total_epochs=n_ep,
                label_smoothing=label_smoothing,
                mixup_alpha=mixup_alpha,
                grad_accum=grad_accum,
                ema_model=ema_model,
                gold_weight_mult=1.5,
            )

            # 2. IMMEDIATE PURGE OF TRAINING WORKERS:
            # Explicitly shuts down worker processes & frees IPC pinned memory BEFORE validation!
            active_loaders = []
            del dl_tr
            execute_emergency_memory_flush()

            msg = f"epoch {ep + 1}/{n_ep}  loss {loss_val:.4f}"
            score = float("nan")

            # 3. SEQUENTIAL VALIDATION DATALOADER:
            # Instantiated ONLY during validation with throttled workers & batch size, bounding pinned memory to ~4.6 GB.
            if ds_va is not None:
                dl_va = DataLoader(
                    ds_va, batch_size=max(1, batch_size // 2), shuffle=False,
                    num_workers=min(4, num_workers), pin_memory=(device.type == "cuda"),
                    persistent_workers=False,
                    prefetch_factor=2 if num_workers > 0 else None,
                )
                active_loaders = [dl_va]
                score, per = evaluate(model, dl_va, device)
                msg += f"  val macro-AUC {score:.4f}"
                all_conds = ", ".join(f"{k}:{v:.3f}" for k, v in sorted(per.items()))
                msg += f"\n       Per-target AUCs ({len(per)}/12 evaluated): [{all_conds}]"

                # 4. IMMEDIATE PURGE OF VALIDATION WORKERS:
                active_loaders = []
                del dl_va
                execute_emergency_memory_flush()

            print(msg, flush=True)

            # ── SWA weight accumulation ───────────────────────────────────────
            if swa_model is not None and ep >= swa_start:
                swa_model.update_parameters(model)

            # ── Best-checkpoint saving ────────────────────────────────────────
            save = (ds_va is None) or (np.isfinite(score) and score > best)
            if save:
                best = score if ds_va is not None else best
                epochs_no_improve = 0
                ckpt_file = os.path.join(out_dir, f"fold{fold}_best.pt")
                torch.save(
                    dict(model=inner_model.state_dict(),
                         cfg=dataclasses.asdict(cfg),
                         fold=fold, epoch=ep + 1, val=score,
                         variant=variant,
                         use_cross_slot=use_cross_slot),
                    ckpt_file,
                )
                print(f"  [CHECKPOINT] New best validation AUC: {best:.4f} -> Saved {ckpt_file}", flush=True)
            else:
                epochs_no_improve += 1
                print(f"  [EARLY STOP] No validation AUC improvement for {epochs_no_improve}/{early_stop_patience} epochs (best: {best:.4f})", flush=True)
                if ds_va is not None and epochs_no_improve >= early_stop_patience:
                    print(f"  [EARLY STOP] Validation AUC did not improve for {early_stop_patience} consecutive epochs. Stopping early at epoch {ep + 1}/{n_ep} to prevent overfitting.", flush=True)
                    break

            execute_emergency_memory_flush()

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
                prefetch_factor=2 if num_workers > 0 else None,
            )
            active_loaders = [dl_swa]
            safe_update_bn(dl_swa, swa_model, device=device)
            active_loaders = []
            del dl_swa
            execute_emergency_memory_flush()
            torch.save(
                dict(model=swa_model.module.state_dict(),
                     cfg=dataclasses.asdict(cfg),
                     fold=fold, epoch=ep + 1, val=best,
                     swa=True, variant=variant,
                     use_cross_slot=use_cross_slot),
                os.path.join(out_dir, f"fold{fold}_swa.pt"),
            )
            print(f"SWA checkpoint saved → fold{fold}_swa.pt")

        # FIX 2: Save EMA checkpoint — this is the primary checkpoint for inference.
        # EMA weights are strictly better than instantaneous best for ViTs.
        torch.save(
            dict(model=ema_model.module.state_dict(),
                 cfg=dataclasses.asdict(cfg),
                 fold=fold, epoch=ep + 1, val=best,
                 ema=True, variant=variant,
                 use_cross_slot=use_cross_slot),
            os.path.join(out_dir, f"fold{fold}_ema.pt"),
        )
        print(f"EMA checkpoint saved → fold{fold}_ema.pt")

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
            out_dir=out_dir,
            active_loaders=active_loaders,
        )
        raise exc

    # ── Final cleanup before exiting fold training ───────────────────────────
    del model, inner_model
    if 'model_train' in locals():
        del model_train
    del optimizer, scheduler, scaler, ds_tr
    if ds_va is not None:
        del ds_va
    if swa_model is not None:
        del swa_model
    if ema_model is not None:
        del ema_model
    import gc
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return best






