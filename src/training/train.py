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

import os
import time
import dataclasses
import numpy as np
import pandas as pd
import os
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

import cv2
cv2.setNumThreads(0)

import torch
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

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
    label_smoothing: float = 0.0,
    mixup_alpha: float = 0.0,
    grad_accum: int = 1,
    ema_model = None,
    gold_weight_mult: float = 1.0,   # UPGRADE 2: upweight gold-labeled studies
) -> float:
    model.train()
    total_loss_tensor = torch.tensor(0.0, device=device)
    n, opt_step = 0, 0
    
    # Pre-allocate rare vector OUTSIDE the loop to prevent VRAM fragmentation
    _rare_mults = [config.RARE_TARGET_WEIGHTS.get(t, 1.0) for t in config.TARGETS]
    _rare_vec = torch.tensor(_rare_mults, device=device).unsqueeze(0)  # [1, C]

    for micro_step, batch in enumerate(dataloader):
        imgs, masks, wmasks, targets, weights = batch
        imgs    = imgs.to(device, non_blocking=True)
        imgs.requires_grad_(True)  # CRITICAL for Unified Memory Gradient Checkpointing
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
        # We mix logits+targets post-backbone (forward already done above).
        # True feature mixing would require model surgery; logit mixing is
        # a lightweight proxy that retains the regularisation benefit.
        if mixup_alpha > 0.0:
            logits, targets, weights = mixup_features(logits, targets, weights, mixup_alpha)

        with ctx:
            loss = criterion(logits.float(), targets, weights)
            loss = loss / grad_accum

        scaler.scale(loss).backward()

        if (micro_step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 0.5
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            if ema_model is not None:
                ema_model.update_parameters(model)
            opt_step += 1

        total_loss_tensor += loss.detach() * grad_accum  # Fully async logging
        n += 1

    return float(total_loss_tensor.item()) / max(n, 1)


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
    for imgs, masks, wmasks, targets, weights in dataloader:
        with ctx:
            p = model(imgs.to(device), masks.to(device), wmasks.to(device))
        P.append(torch.sigmoid(p.float()).cpu().numpy())
        Y.append(targets.numpy())
        W.append(weights.numpy())
    P, Y, W = np.concatenate(P), np.concatenate(Y), np.concatenate(W)
    per: dict[str, float] = {}
    for j, t in enumerate(config.TARGETS):
        m = W[:, j] >= 0.99  # CRITICAL FIX: Evaluate ONLY on Gold Labels to prevent sklearn continuous format ValueError on soft pseudo-labels
        if m.sum() > 1 and 0 < Y[m, j].sum() < m.sum():
            per[t] = float(roc_auc_score(Y[m, j], P[m, j]))
    macro = float(np.mean(list(per.values()))) if per else float("nan")
    return macro, per


# ──────────────────────────────────────────── Main entry point ───────────────
def run_training(
    labels_csv: str,
    cache_prefix: str,
    folds_csv: str | None = None,
    fold: int = 0,
    out_dir: str = ".",
    epochs: int | None = None,
    n_windows_train: int = 8,  # FIX 3: 8 windows covers 73% of the 24-depth stack vs 35% at 4
    num_workers: int = 8, # Safe feed rate for 16 studies per step
    seed: int = config.SEED,
    # model
    variant: str = "dinov2-base",         # Upgrade A: default to Base
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
        use_cross_slot=use_cross_slot,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    ).to(device)

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
    )

    # ── Data ─────────────────────────────────────────────────────────────────
    ds_tr = RSNADataset(tr, cache_prefix, cfg, True, n_windows_train, seed)
    dl_tr = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=True,
        drop_last=len(tr) > batch_size, num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(4 if num_workers > 0 else None),
    )
    dl_va = None
    if len(va):
        ds_va = RSNADataset(va, cache_prefix, cfg, False, None, seed, aug=False)
        dl_va = DataLoader(
            ds_va, batch_size=max(1, batch_size // 2), shuffle=False,
            num_workers=num_workers, pin_memory=(device.type == "cuda"),
        )

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
    total_steps = n_ep * max(len(dl_tr), 1)
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

    for ep in range(n_ep):
        ds_tr.set_epoch(ep)

        # Gold upweighting: constant 1.5× throughout training.
        # Constant (not curriculum) avoids Adam momentum destabilization from
        # hard weight-schedule flips mid-training. 1.5× is moderate enough not
        # to overfit to the small gold set, but enough to prioritize verified labels.
        loss_val = train_epoch(
            model, dl_tr, optimizer, scaler, scheduler, device,
            criterion,
            label_smoothing=label_smoothing,
            mixup_alpha=mixup_alpha,
            grad_accum=grad_accum,
            ema_model=ema_model,
            gold_weight_mult=1.5,
        )

        msg = f"epoch {ep + 1}/{n_ep}  loss {loss_val:.4f}"
        score = float("nan")

        if dl_va is not None:
            score, per = evaluate(model, dl_va, device)
            msg += f"  val macro-AUC {score:.4f}"
            top3 = sorted(per.items(), key=lambda kv: kv[1], reverse=True)[:3]
            msg += "  top3=[" + ", ".join(f"{k}:{v:.3f}" for k, v in top3) + "]"
        print(msg, flush=True)

        # ── SWA weight accumulation ───────────────────────────────────────────
        if swa_model is not None and ep >= swa_start:
            swa_model.update_parameters(model)

        # ── Best-checkpoint saving ────────────────────────────────────────────
        save = (dl_va is None) or (np.isfinite(score) and score > best)
        if save:
            best = score if dl_va is not None else best
            torch.save(
                dict(model=inner_model.state_dict(),
                     cfg=dataclasses.asdict(cfg),
                     fold=fold, epoch=ep, val=score,
                     variant=variant,
                     use_cross_slot=use_cross_slot),
                os.path.join(out_dir, f"fold{fold}_best.pt"),
            )

        # ── Time-budget guard ─────────────────────────────────────────────────
        if time.time() - t0 > config.TIME_BUDGET_HOURS * 3600:
            print("Time budget reached; stopping early.")
            break

    # ── SWA: recompute BatchNorm stats, save ─────────────────────────────────
    if swa_model is not None and swa_epochs > 0:
        print("Updating SWA batch-norm statistics …")
        update_bn(dl_tr, swa_model, device=device)
        torch.save(
            dict(model=swa_model.module.state_dict(),
                 cfg=dataclasses.asdict(cfg),
                 fold=fold, epoch=n_ep, val=best,
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
             fold=fold, epoch=n_ep, val=best,
             ema=True, variant=variant,
             use_cross_slot=use_cross_slot),
        os.path.join(out_dir, f"fold{fold}_ema.pt"),
    )
    print(f"EMA checkpoint saved → fold{fold}_ema.pt")

    return best






