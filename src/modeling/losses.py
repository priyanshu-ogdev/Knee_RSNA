"""Loss functions for multi-label training.

Implementations
---------------
AsymmetricLoss  Ridnik et al., ICCV 2021 (arXiv:2009.14119).
                Uses separate positive and negative focusing exponents and an
                optional negative-probability clip. Its effect on this dataset
                must be compared empirically with BCE.

                gamma_neg may be shared across targets or supplied per target.
                A larger negative focusing exponent suppresses easy negative
                examples; it does not directly upweight positive examples.

WeightedBCE     Legacy for A/B ablation; kept as a drop-in alternative.

Reference: https://github.com/Alibaba-MIIL/ASL
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """Ridnik et al., ICCV 2021.

    Parameters
    ----------
    gamma_neg : float or list[float]
        Focusing exponent for negatives.  Scalar: applied uniformly across all
        classes.  List[C]: per-target gamma, enabling prevalence-matched focusing
        (higher gamma for rare targets, lower for common ones).
    gamma_pos : float
        Focusing exponent for positives. Default 0 — never down-weight true
        positives (critical for rare, high-stakes findings).
    clip : float
        Probability shift for negatives. Entries where p_neg < clip are zeroed,
        discarding trivially easy / likely mislabelled negatives.
        Set clip=0.0 to disable (degenerates to asymmetric Focal Loss).
    eps : float
        Log-stability epsilon.
    """

    def __init__(self, gamma_neg=2.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        if isinstance(gamma_neg, (list, tuple)):
            # Register as buffer so it moves to GPU with .to(device) automatically
            gamma_tensor = torch.tensor(gamma_neg, dtype=torch.float32)
            if not torch.isfinite(gamma_tensor).all() or (gamma_tensor < 0).any():
                raise ValueError("gamma_neg values must be finite and non-negative")
            self.register_buffer("gamma_neg_vec", gamma_tensor)
            self._gamma_neg_scalar = None
        else:
            if not torch.isfinite(torch.tensor(float(gamma_neg))) or gamma_neg < 0:
                raise ValueError("gamma_neg must be finite and non-negative")
            self.gamma_neg_vec = None
            self._gamma_neg_scalar = float(gamma_neg)
        if not torch.isfinite(torch.tensor(float(gamma_pos))) or gamma_pos < 0:
            raise ValueError("gamma_pos must be finite and non-negative")
        if not torch.isfinite(torch.tensor(float(clip))) or not 0 <= clip < 1:
            raise ValueError("clip must be finite and in [0, 1)")
        self.gamma_pos = float(gamma_pos)
        self.clip = float(clip)
        self.eps = eps

    # ------------------------------------------------------------------
    def forward(
        self,
        logits: torch.Tensor,          # [B, C] raw (pre-sigmoid)
        targets: torch.Tensor,         # [B, C] in [0, 1]; soft labels OK
        weights: torch.Tensor | None = None,  # [B, C] per-entry weight
    ) -> torch.Tensor:
        # Masked labels may be represented as NaN by upstream dataframes.
        # Replace them before arithmetic; the weight mask still controls their
        # contribution and prevents NaN * 0 propagation.
        targets = torch.nan_to_num(targets, nan=0.0).clamp(0.0, 1.0)
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # Probability shift — kills easy negatives & mislabelled entries
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Log loss terms
        loss_pos = targets       * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg   # [B, C]

        # Asymmetric focusing — down-weight easy examples per class
        # gamma_neg: scalar or [C] tensor (broadcasts correctly either way)
        # SOTA Bugfix: ensure g_neg always matches targets.device and targets.dtype
        if self.gamma_neg_vec is not None:
            if self.gamma_neg_vec.numel() != logits.shape[1]:
                raise ValueError(
                    f"gamma_neg has {self.gamma_neg_vec.numel()} values for "
                    f"{logits.shape[1]} targets"
                )
            g_neg = self.gamma_neg_vec.to(device=targets.device, dtype=targets.dtype)
        else:
            g_neg = logits.new_full((1,), self._gamma_neg_scalar, device=targets.device, dtype=targets.dtype)
        pt    = xs_pos * targets + xs_neg * (1.0 - targets)   # [B, C]
        gamma = self.gamma_pos * targets + g_neg * (1.0 - targets)  # [B, C]
        loss  = loss * ((1.0 - pt) ** gamma)

        loss = -loss  # minimise

        if weights is not None:
            if weights.shape != loss.shape:
                raise ValueError(
                    f"weights shape {tuple(weights.shape)} must match loss shape "
                    f"{tuple(loss.shape)}"
                )
            loss = loss * weights
            # Normalize by labeled entries, not confidence-weight sum: confidence
            # then scales the loss as intended rather than canceling itself out.
            labeled_count = (weights > 0).sum(dim=0).clamp_min(1)
            class_losses = loss.sum(dim=0) / labeled_count
            return class_losses.mean()
        return loss.mean()


class WeightedBCE(nn.Module):
    """Legacy weighted BCE — kept for ablation comparisons."""

    def forward(self, logits, targets, weights=None):
        loss = F.binary_cross_entropy_with_logits(logits.float(), targets,
                                                  reduction="none")
        if weights is not None:
            if weights.shape != loss.shape:
                raise ValueError(
                    f"weights shape {tuple(weights.shape)} must match loss shape "
                    f"{tuple(loss.shape)}"
                )
            loss = loss * weights
            labeled_count = (weights > 0).sum(dim=0).clamp_min(1)
            class_losses = loss.sum(dim=0) / labeled_count
            return class_losses.mean()
        return loss.mean()


class PairwiseAUC(nn.Module):
    """Dense, masked pairwise ranking loss aligned with ROC-AUC."""

    def __init__(self, margin: float = 0.0):
        super().__init__()
        self.margin = float(margin)

    def forward(self, logits, targets, weights=None):
        if logits.ndim != 2 or targets.shape != logits.shape:
            raise ValueError("logits and targets must both have shape [B, C]")
        # NaN is the label-engine's explicit "unknown" value.  It must never
        # enter the negative pool (otherwise weak-label masking silently turns
        # missing labels into negatives).
        valid = torch.isfinite(targets)
        if weights is not None:
            if weights.shape != targets.shape:
                raise ValueError("weights must match logits shape")
            valid = valid & torch.isfinite(weights) & (weights > 0)
        terms = []
        for c in range(logits.shape[1]):
            pos = logits[valid[:, c] & (targets[:, c] >= 0.5), c]
            neg = logits[valid[:, c] & (targets[:, c] < 0.5), c]
            if pos.numel() and neg.numel():
                pair_loss = torch.nn.functional.softplus(
                    self.margin - pos[:, None] + neg[None, :]
                )
                # Confidence weights affect pair contribution without
                # changing the ranking target.  Normalize by total pair
                # weight so the scale is stable across batches.
                if weights is not None:
                    pos_w = weights[valid[:, c] & (targets[:, c] >= 0.5), c]
                    neg_w = weights[valid[:, c] & (targets[:, c] < 0.5), c]
                    pair_w = pos_w[:, None] * neg_w[None, :]
                    terms.append((pair_loss * pair_w).sum() / pair_w.sum().clamp_min(1e-8))
                else:
                    terms.append(pair_loss.mean())
        return torch.stack(terms).mean() if terms else logits.sum() * 0.0


class AUCAsymmetricLoss(nn.Module):
    """ASL plus a small pairwise term that directly optimises ranking."""

    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05,
                 rank_weight: float = 0.15, margin: float = 0.0):
        super().__init__()
        if not 0.0 <= rank_weight <= 1.0:
            raise ValueError("rank_weight must be in [0, 1]")
        self.rank_weight = float(rank_weight)
        self.asl = AsymmetricLoss(gamma_neg, gamma_pos, clip)
        self.rank = PairwiseAUC(margin)

    def forward(self, logits, targets, weights=None):
        return ((1.0 - self.rank_weight) * self.asl(logits, targets, weights)
                + self.rank_weight * self.rank(logits, targets, weights))


class LocalAttentionSmoothness(nn.Module):
    """Penalize abrupt attention changes along adjacent valid windows.

    The module is intentionally separate from the classifier loss: callers can
    apply it to exposed attention maps without forcing attention tensors into
    the checkpoint/inference contract.  ``attention`` may be [B,S,W,C] or
    [B,S,W], and ``mask`` must be [B,S,W].
    """

    def forward(self, attention, mask):
        if attention.ndim not in (3, 4) or mask.ndim != 3:
            raise ValueError("attention must be [B,S,W,(C)] and mask [B,S,W]")
        if attention.shape[:3] != mask.shape:
            raise ValueError("attention and mask window dimensions must match")
        delta = attention[:, :, 1:] - attention[:, :, :-1]
        pair_mask = (mask[:, :, 1:] > 0.5) & (mask[:, :, :-1] > 0.5)
        if attention.ndim == 4:
            pair_mask = pair_mask.unsqueeze(-1)
        values = delta.square().masked_select(pair_mask)
        return values.mean() if values.numel() else attention.sum() * 0.0


LOSS_REGISTRY: dict[str, type] = {
    "asl": AsymmetricLoss,
    "bce": WeightedBCE,
    "auc_asl": AUCAsymmetricLoss,
}


def build_loss(name: str = "asl", **kwargs) -> nn.Module:
    """Factory.  name in {'asl', 'bce'}.  Extra kwargs forwarded to __init__."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**kwargs)
