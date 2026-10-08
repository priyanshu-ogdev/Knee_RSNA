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


LOSS_REGISTRY: dict[str, type] = {"asl": AsymmetricLoss, "bce": WeightedBCE}


def build_loss(name: str = "asl", **kwargs) -> nn.Module:
    """Factory.  name in {'asl', 'bce'}.  Extra kwargs forwarded to __init__."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**kwargs)
