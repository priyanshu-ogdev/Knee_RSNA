"""Training loop on the memmap cache (data-contract fixes only; architecture/recipe unchanged).

Fixed vs the baseline: TIME_BUDGET_HOURS exists; unlabeled targets are masked (no NaN loss); loss is normalised by
the weight mass; bf16 autocast where supported (fp16+GradScaler otherwise); seeded; fold-aware validation with
per-label AUC; best-checkpoint saving; epoch-seeded augmentation.
"""
import os
import time
import dataclasses
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from . import config
from .model import build_model
from .dataset import RSNADataset
from .preprocess.cache import cfg_of as cache_cfg


def _autocast(device):
    if device.type != 'cuda':
        return torch.autocast('cpu', enabled=False), False
    use_bf16 = torch.cuda.is_bf16_supported()
    return torch.autocast('cuda', dtype=torch.bfloat16 if use_bf16 else torch.float16), (not use_bf16)


def train_epoch(model, dataloader, optimizer, scaler, scheduler, device):
    model.train()
    total, n = 0.0, 0
    for imgs, masks, wmasks, targets, weights in dataloader:
        imgs, masks, wmasks = imgs.to(device, non_blocking=True), masks.to(device), wmasks.to(device)
        targets, weights = targets.to(device), weights.to(device)
        ctx, _ = _autocast(device)
        with ctx:
            preds = model(imgs, masks, wmasks)
        loss = (F.binary_cross_entropy_with_logits(preds.float(), targets, reduction='none') * weights).sum() \
            / weights.sum().clamp_min(1.0)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += float(loss.item())
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, dataloader, device):
    """-> (macro AUC over labels that have both classes, {label: auc}); weights==0 entries are ignored."""
    from sklearn.metrics import roc_auc_score
    model.eval()
    P, Y, W = [], [], []
    for imgs, masks, wmasks, targets, weights in dataloader:
        ctx, _ = _autocast(device)
        with ctx:
            p = model(imgs.to(device), masks.to(device), wmasks.to(device))
        P.append(torch.sigmoid(p.float()).cpu().numpy())
        Y.append(targets.numpy())
        W.append(weights.numpy())
    P, Y, W = np.concatenate(P), np.concatenate(Y), np.concatenate(W)
    per = {}
    for j, t in enumerate(config.TARGETS):
        m = W[:, j] > 0
        if m.sum() > 1 and 0 < Y[m, j].sum() < m.sum():
            per[t] = float(roc_auc_score(Y[m, j], P[m, j]))
    return (float(np.mean(list(per.values()))) if per else float('nan')), per


def run_training(labels_csv, cache_prefix, folds_csv=None, fold=0, out_dir='.', epochs=None,
                 n_windows_train=4, num_workers=4, seed=config.SEED, variant='dinov2-small'):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = cache_cfg(cache_prefix)           # the config the cache was BUILT with (never re-derive it from a preset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_csv(labels_csv)
    if folds_csv:
        df = df.merge(pd.read_csv(folds_csv)[['StudyInstanceUID', 'fold']], on='StudyInstanceUID', how='inner')
        tr, va = df[df.fold != fold], df[df.fold == fold]
    else:
        tr, va = df, df.iloc[:0]
    print(f'device={device} train={len(tr)} val={len(va)} preset={cfg.name} D={cfg.stack_depth} img={cfg.img_size}')
    model = build_model(variant=variant).to(device)      # variant may be a local folder (offline Kaggle dataset)
    ds_tr = RSNADataset(tr, cache_prefix, cfg, True, n_windows_train, seed)
    dl_tr = DataLoader(ds_tr, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=len(tr) > config.BATCH_SIZE,
                       num_workers=num_workers, pin_memory=device.type == 'cuda',
                       persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers > 0 else None)
    dl_va = None
    if len(va):
        ds_va = RSNADataset(va, cache_prefix, cfg, False, None, seed, aug=False)
        dl_va = DataLoader(ds_va, batch_size=max(1, config.BATCH_SIZE // 2), shuffle=False, num_workers=num_workers,
                           pin_memory=device.type == 'cuda')
    opt = torch.optim.AdamW([
        {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': config.LR_BACKBONE},
        {'params': list(model.wpool.parameters()) + list(model.head.parameters()), 'lr': config.LR_HEAD},
    ], weight_decay=config.WEIGHT_DECAY)
    n_ep = epochs or config.EPOCHS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[config.LR_BACKBONE, config.LR_HEAD],
                                                total_steps=n_ep * max(len(dl_tr), 1), pct_start=0.15)
    _, need_scaler = _autocast(device)
    scaler = torch.amp.GradScaler('cuda', enabled=need_scaler)
    os.makedirs(out_dir, exist_ok=True)
    best, t0 = -1.0, time.time()
    for ep in range(n_ep):
        ds_tr.set_epoch(ep)
        loss = train_epoch(model, dl_tr, opt, scaler, sched, device)
        msg = f'epoch {ep + 1}/{n_ep} loss {loss:.4f}'
        score = float('nan')
        if dl_va is not None:
            score, per = evaluate(model, dl_va, device)
            msg += f' | val macro-AUC {score:.4f}'
        print(msg, flush=True)
        if dl_va is None or (np.isfinite(score) and score > best):
            best = score if dl_va is not None else best
            torch.save(dict(model=model.state_dict(), cfg=dataclasses.asdict(cfg), fold=fold, epoch=ep, val=score),
                       os.path.join(out_dir, f'fold{fold}_best.pt'))
        if time.time() - t0 > config.TIME_BUDGET_HOURS * 3600:
            print('Time budget reached; stopping early.')
            break
    return best
