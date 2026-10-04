"""Loss functions for multi-label training.

Implementations
---------------
AsymmetricLoss  Ridnik et al., ICCV 2021 (arXiv:2009.14119).
                De-facto SOTA for multi-label classification.
                Outperforms BCE on MS-COCO (+2.5pp mAP), Open Images (+1.4pp),
                and Focal Loss on every tested benchmark.
                Mechanism: decoupled gamma+/gamma- focusing + probability-shift
                that discards easy/mislabelled negatives (directly relevant to our
                NLP-derived noisy labels where some negative labels are unreliable).

                UPGRADE (v3): gamma_neg now accepts a per-target list/tensor [C].
                Using per-target gammas aligns the negative-suppression strength
                with empirical label prevalence:
                  - Rare positives (MCL 15%, Baker's 21%) get high gamma_neg (4.0)
                    so the loss focuses hard on their rare positive samples.
                  - Common positives (Effusion 60%) get low gamma_neg (1.0)
                    so we do not suppress the informative negative gradient.

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
            self.register_buffer(
                "gamma_neg_vec",
                torch.tensor(gamma_neg, dtype=torch.float32)
            )
            self._gamma_neg_scalar = None
        else:
            self.gamma_neg_vec = None
            self._gamma_neg_scalar = float(gamma_neg)
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
        xs_neg = xs_pos

        # Probability shift — kills easy negatives & mislabelled entries
        if self.clip > 0:
            xs_neg = (xs_neg - self.clip).clamp(min=0.0)

        # Log loss terms
        loss_pos = targets       * torch.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1.0 - targets) * torch.log((1.0 - xs_neg).clamp(min=self.eps))
        loss = loss_pos + loss_neg   # [B, C]

        # Asymmetric focusing — down-weight easy examples per class
        # gamma_neg: scalar or [C] tensor (broadcasts correctly either way)
        g_neg = (self.gamma_neg_vec
                 if self.gamma_neg_vec is not None
                 else logits.new_full((1,), self._gamma_neg_scalar))
        pt    = xs_pos * targets + xs_neg * (1.0 - targets)   # [B, C]
        gamma = self.gamma_pos * targets + g_neg * (1.0 - targets)  # [B, C]
        loss  = loss * ((1.0 - pt) ** gamma)

        loss = -loss  # minimise

        if weights is not None:
            loss = loss * weights
            # Compute loss independently per class first, then average across classes
            # This prevents a rare-target's massive weight from shrinking the loss gradients of common targets
            class_losses = loss.sum(dim=0) / weights.sum(dim=0).clamp_min(1.0)
            return class_losses.mean()
        return loss.mean()


class WeightedBCE(nn.Module):
    """Legacy weighted BCE — kept for ablation comparisons."""

    def forward(self, logits, targets, weights=None):
        loss = F.binary_cross_entropy_with_logits(logits.float(), targets,
                                                  reduction="none")
        if weights is not None:
            loss = loss * weights
            # Compute loss independently per class first, then average across classes
            # This prevents a rare-target's massive weight from shrinking the loss gradients of common targets
            class_losses = loss.sum(dim=0) / weights.sum(dim=0).clamp_min(1.0)
            return class_losses.mean()
        return loss.mean()


LOSS_REGISTRY: dict[str, type] = {"asl": AsymmetricLoss, "bce": WeightedBCE}


def build_loss(name: str = "asl", **kwargs) -> nn.Module:
    """Factory.  name in {'asl', 'bce'}.  Extra kwargs forwarded to __init__."""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(LOSS_REGISTRY)}")
    return LOSS_REGISTRY[name](**kwargs)
